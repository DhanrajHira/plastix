// No include guards -- designed for multiple inclusion (one per pass).
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
