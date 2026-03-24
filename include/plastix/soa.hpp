// No include guards — designed for multiple inclusion (one per pass).
//
// This file is an X-macro code generator. It is included three times (by
// unit_state.hpp, or any file that defines SOA types) with a different
// PLASTIX_SOA_MODE_* macro set each time. Each inclusion expands the same
// .inc field list into a different set of declarations:
//
//   PLASTIX_SOA_MODE_TAGS  → forward-declares the entity struct and all field
//                            tag types (e.g. struct ActivationATag {}).
//
//   PLASTIX_SOA_MODE_ALLOC → emits a SOAAllocator typedef that bundles all
//                            declared fields (e.g. UnitStateAllocator).
//
//   PLASTIX_SOA_MODE_HANDLE → emits a Handle struct with typed Get* accessors
//                             for each field, wrapping a raw AllocId.
//
// Adding a new SOA type:
//   1. Create a MyType.inc file with SOA_TYPE / FIELD / SOA_END macros.
//   2. Include soa.hpp three times with the three mode guards, just as
//      unit_state.hpp does for unit_state.inc.
#include "plastix/alloc.hpp"

#undef SOA_TYPE
#undef FIELD
#undef SOA_END

#if defined(PLASTIX_SOA_MODE_TAGS)
#undef PLASTIX_SOA_MODE_TAGS

// Emits: struct Name {}; using NameId = AllocId<Name>;
//        struct FieldNameTag {};  (one per FIELD)
#define SOA_TYPE(Name)                                                         \
  struct Name {};                                                              \
  using Name##Id = plastix::alloc::AllocId<Name>;
#define FIELD(Name, Type)                                                      \
  struct Name##Tag {};
#define SOA_END()

#elif defined(PLASTIX_SOA_MODE_ALLOC)
#undef PLASTIX_SOA_MODE_ALLOC

// Emits: using NameAllocator = SOAAllocator<Name, SOAField<Tag, Type>, ...>;
#define SOA_TYPE(Name)                                                         \
  using Name##Allocator = plastix::alloc::SOAAllocator < Name
#define FIELD(Name, Type) , plastix::alloc::SOAField<Name##Tag, Type>
#define SOA_END() > ;

#elif defined(PLASTIX_SOA_MODE_HANDLE)
#undef PLASTIX_SOA_MODE_HANDLE

// Emits: struct NameHandle { ...; Type &GetFieldName(Alloc_ &A); ... };
// The handle stores an AllocId and provides named accessors so call sites
// can write handle.GetActivationA(alloc) instead of alloc.Get<Tag>(id).
#define SOA_TYPE(Name)                                                         \
  struct Name##Handle {                                                        \
    using Alloc_ = Name##Allocator;                                            \
    plastix::alloc::AllocId<Name> Id;                                          \
    Name##Handle(plastix::alloc::AllocId<Name> Id) : Id(Id) {}                 \
    operator plastix::alloc::AllocId<Name>() const { return Id; }
#define FIELD(Name, Type)                                                      \
  Type &Get##Name(Alloc_ &A) { return A.Get<Name##Tag>(Id); }
#define SOA_END()                                                              \
  }                                                                            \
  ;

#endif
