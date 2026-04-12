// Streaming iPC — MNIST Classification (v3)
//
// Three-phase predictive coding on MNIST:
//   1. Inference: input clamped, output free -> argmax prediction
//   2. Learning:  input + output clamped -> Hebbian weight updates
//   3. Rest:      zero input, output free -> network relaxes
//
// Usage: ./ipc_multilayer_v3 <images> <labels> [max-samples] [alpha]

#include <plastix/plastix.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// --- MNIST loader ---

static uint32_t ReadU32BE(std::ifstream &F) {
  uint8_t B[4];
  F.read(reinterpret_cast<char *>(B), 4);
  return (uint32_t(B[0]) << 24) | (uint32_t(B[1]) << 16) |
         (uint32_t(B[2]) << 8) | uint32_t(B[3]);
}

struct MNISTData {
  std::vector<float> Images;
  std::vector<uint8_t> Labels;
  size_t N = 0, ImgSize = 0;
  const float *Image(size_t I) const { return &Images[I * ImgSize]; }
};

static MNISTData LoadMNIST(const std::string &ImgPath,
                           const std::string &LblPath) {
  MNISTData D;
  std::ifstream ImgF(ImgPath, std::ios::binary);
  if (!ImgF) { std::cerr << "Cannot open " << ImgPath << "\n"; std::exit(1); }
  if (ReadU32BE(ImgF) != 2051) { std::cerr << "Bad image magic\n"; std::exit(1); }
  D.N = ReadU32BE(ImgF);
  uint32_t Rows = ReadU32BE(ImgF), Cols = ReadU32BE(ImgF);
  D.ImgSize = Rows * Cols;
  std::vector<uint8_t> Raw(D.N * D.ImgSize);
  ImgF.read(reinterpret_cast<char *>(Raw.data()), Raw.size());
  D.Images.resize(Raw.size());
  for (size_t I = 0; I < Raw.size(); ++I)
    D.Images[I] = Raw[I] / 255.0f;

  std::ifstream LblF(LblPath, std::ios::binary);
  if (!LblF) { std::cerr << "Cannot open " << LblPath << "\n"; std::exit(1); }
  if (ReadU32BE(LblF) != 2049) { std::cerr << "Bad label magic\n"; std::exit(1); }
  if (ReadU32BE(LblF) != D.N) { std::cerr << "Count mismatch\n"; std::exit(1); }
  D.Labels.resize(D.N);
  LblF.read(reinterpret_cast<char *>(D.Labels.data()), D.N);
  return D;
}

// --- Extra per-unit fields ---

struct ErrorTag {};
struct BottomUpTag {};

// --- Topology & hyperparameters ---

constexpr size_t NumInputs = 784;
constexpr size_t HiddenSize = 128;
constexpr size_t NumOutputs = 10;
constexpr size_t HiddenBegin = NumInputs;
constexpr size_t HiddenEnd = HiddenBegin + HiddenSize;
constexpr size_t OutputBegin = HiddenEnd;
constexpr size_t OutputEnd = OutputBegin + NumOutputs;

float Alpha = 0.001f;
float Gamma = 0.5f;

constexpr size_t InferenceSteps = 10;
constexpr size_t LearningSteps = 10;
constexpr size_t RestSteps = 10;

// --- Activation (ReLU, applied only to hidden sources) ---

inline float Act(float X) { return X > 0.0f ? X : 0.0f; }
inline float ActDeriv(float X) { return X > 0.0f ? 1.0f : 0.0f; }

inline bool IsHidden(size_t Id) { return Id >= HiddenBegin && Id < HiddenEnd; }
inline float ActSrc(float X, size_t SrcId) { return IsHidden(SrcId) ? Act(X) : X; }

// --- iPC policies ---

struct iPCForwardPass {
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t SrcId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    float W = C.template Get<plastix::ConnPageMarker>(PageId).GetSlot(SlotIdx).second;
    return W * ActSrc(U.template Get<plastix::ActivationTag>(SrcId), SrcId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Mu) {
    U.template Get<ErrorTag>(Id) =
        U.template Get<plastix::ActivationTag>(Id) - Mu;
  }
};

struct iPCBackwardPass {
  using Accumulator = float;
  static float Map(auto &U, size_t, size_t DstId, auto &C, size_t PageId,
                   size_t SlotIdx, auto &) {
    float W = C.template Get<plastix::ConnPageMarker>(PageId).GetSlot(SlotIdx).second;
    return W * U.template Get<ErrorTag>(DstId);
  }
  static float Combine(float A, float B) { return A + B; }
  static void Apply(auto &U, size_t Id, auto &, float Acc) {
    if (IsHidden(Id))
      U.template Get<BottomUpTag>(Id) =
          ActDeriv(U.template Get<plastix::ActivationTag>(Id)) * Acc;
  }
};

struct iPCUpdateConn {
  static void UpdateIncomingConnection(auto &U, size_t DstId, size_t SrcId,
                                       auto &C, size_t PageId, size_t SlotIdx, auto &) {
    float Eps = U.template Get<ErrorTag>(DstId);
    float Fx = ActSrc(U.template Get<plastix::ActivationTag>(SrcId), SrcId);
    C.template Get<plastix::ConnPageMarker>(PageId).Conn[SlotIdx].second += Alpha * Eps * Fx;
  }
  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

struct iPCUpdateUnit {
  static void Update(auto &U, size_t Id, auto &) {
    if (IsHidden(Id)) {
      float Eps = U.template Get<ErrorTag>(Id);
      float BU = U.template Get<BottomUpTag>(Id);
      U.template Get<plastix::ActivationTag>(Id) += Gamma * (-Eps + BU);
      U.template Get<BottomUpTag>(Id) = 0.0f;
    } else if (Id >= OutputBegin && Id < OutputEnd) {
      U.template Get<plastix::ActivationTag>(Id) +=
          Gamma * (-U.template Get<ErrorTag>(Id));
    }
  }
};

// --- Traits & types ---

struct iPCTraits : plastix::DefaultNetworkTraits<plastix::ConnStateAllocator> {
  using ForwardPass = iPCForwardPass;
  using BackwardPass = iPCBackwardPass;
  using UpdateUnit = iPCUpdateUnit;
  using UpdateConn = iPCUpdateConn;
  using ExtraUnitFields = plastix::UnitFieldList<
      plastix::alloc::SOAField<ErrorTag, float>,
      plastix::alloc::SOAField<BottomUpTag, float>>;
};

using iPCNet = plastix::Network<iPCTraits>;
using FC = plastix::FullyConnected;

// --- iPC step ---
// ClampTarget=true: clamp output to label, update weights (learning)
// ClampTarget=false: output free, no weight update (inference / rest)

void iPCStep(iPCNet &Net, std::span<const float> Input, uint8_t Label,
             bool ClampTarget) {
  if (ClampTarget) {
    auto &U = Net.GetUnitAlloc();
    for (size_t I = 0; I < NumOutputs; ++I)
      U.Get<plastix::ActivationTag>(OutputBegin + I) = (I == Label) ? 1.0f : 0.0f;
  }

  Net.DoForwardPass(Input);
  Net.DoBackwardPass();
  if (ClampTarget)
    Net.DoUpdateConnectionState();
  Net.DoUpdateUnitState();
}

static size_t Argmax(iPCNet &Net) {
  auto &U = Net.GetUnitAlloc();
  size_t Best = 0;
  float BestVal = U.Get<plastix::ActivationTag>(OutputBegin);
  for (size_t I = 1; I < NumOutputs; ++I) {
    float V = U.Get<plastix::ActivationTag>(OutputBegin + I);
    if (V > BestVal) { BestVal = V; Best = I; }
  }
  return Best;
}

// --- Main ---

int main(int argc, char *argv[]) {
  std::string ImgPath = "train-images-idx3-ubyte";
  std::string LblPath = "train-labels-idx1-ubyte";
  size_t MaxSamples = 0;

  if (argc >= 3) { ImgPath = argv[1]; LblPath = argv[2]; }
  if (argc >= 4) MaxSamples = std::stoul(argv[3]);
  if (argc >= 5) Alpha = std::stof(argv[4]);

  std::cout << "Plastix iPC MNIST (v3)\n======================\n";
  auto Data = LoadMNIST(ImgPath, LblPath);
  size_t NumSamples = (MaxSamples > 0 && MaxSamples < Data.N) ? MaxSamples : Data.N;

  std::cout << "Loaded " << Data.N << " images, using " << NumSamples << "\n"
            << "Network: " << NumInputs << " -> " << HiddenSize << " -> " << NumOutputs << "\n"
            << "Steps: infer=" << InferenceSteps << " learn=" << LearningSteps
            << " rest=" << RestSteps << "\n"
            << "Alpha=" << Alpha << " Gamma=" << Gamma << "\n" << std::endl;

  iPCNet Net(NumInputs, FC{HiddenSize, 0.0f}, FC{NumOutputs, 0.0f});

  // LeCun-uniform weight init
  std::mt19937 Rng(42);
  {
    float BH = std::sqrt(3.0f / NumInputs), BO = std::sqrt(3.0f / HiddenSize);
    std::uniform_real_distribution<float> HD(-BH, BH), OD(-BO, BO);
    auto &CA = Net.GetConnAlloc();
    for (size_t P = 0; P < CA.Size(); ++P) {
      auto &Page = CA.Get<plastix::ConnPageMarker>(P);
      for (size_t S = 0; S < Page.Count; ++S)
        Page.Conn[S].second = (Page.ToUnitIdx >= OutputBegin) ? OD(Rng) : HD(Rng);
    }
  }

  std::vector<float> ZeroInput(NumInputs, 0.0f);
  size_t Correct = 0, Total = 0, WinCorrect = 0, WinTotal = 0;
  constexpr size_t PrintEvery = 1000;
  std::cout << std::fixed << std::setprecision(4);

  for (size_t S = 0; S < NumSamples; ++S) {
    std::span<const float> Img(Data.Image(S), Data.ImgSize);
    uint8_t Label = Data.Labels[S];

    for (size_t T = 0; T < InferenceSteps; ++T)
      iPCStep(Net, Img, Label, false);

    bool Hit = (Argmax(Net) == Label);
    Correct += Hit; ++Total; WinCorrect += Hit; ++WinTotal;

    for (size_t T = 0; T < LearningSteps; ++T)
      iPCStep(Net, Img, Label, true);

    for (size_t T = 0; T < RestSteps; ++T)
      iPCStep(Net, ZeroInput, Label, false);

    if ((S + 1) % PrintEvery == 0) {
      std::cout << "Sample " << std::setw(6) << S + 1
                << "  window: " << float(WinCorrect) / WinTotal
                << "  cumulative: " << float(Correct) / Total << std::endl;
      WinCorrect = WinTotal = 0;
    }
  }

  std::cout << "\nFinal prequential accuracy: " << float(Correct) / Total
            << "  (" << Correct << "/" << Total << ")" << std::endl;
  return 0;
}
