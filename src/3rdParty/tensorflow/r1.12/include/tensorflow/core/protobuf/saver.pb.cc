// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/saver.proto

#include "tensorflow/core/protobuf/saver.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace tensorflow {
class SaverDefDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<SaverDef>
      _instance;
} _SaverDef_default_instance_;
}  // namespace tensorflow
namespace protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto {
static void InitDefaultsSaverDef() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_SaverDef_default_instance_;
    new (ptr) ::tensorflow::SaverDef();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::SaverDef::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_SaverDef =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsSaverDef}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_SaverDef.base);
}

::google::protobuf::Metadata file_level_metadata[1];
const ::google::protobuf::EnumDescriptor* file_level_enum_descriptors[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, filename_tensor_name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, save_tensor_name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, restore_op_name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, max_to_keep_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, sharded_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, keep_checkpoint_every_n_hours_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::SaverDef, version_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::SaverDef)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::tensorflow::_SaverDef_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "tensorflow/core/protobuf/saver.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, file_level_enum_descriptors, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n$tensorflow/core/protobuf/saver.proto\022\n"
      "tensorflow\"\236\002\n\010SaverDef\022\034\n\024filename_tens"
      "or_name\030\001 \001(\t\022\030\n\020save_tensor_name\030\002 \001(\t\022"
      "\027\n\017restore_op_name\030\003 \001(\t\022\023\n\013max_to_keep\030"
      "\004 \001(\005\022\017\n\007sharded\030\005 \001(\010\022%\n\035keep_checkpoin"
      "t_every_n_hours\030\006 \001(\002\022=\n\007version\030\007 \001(\0162,"
      ".tensorflow.SaverDef.CheckpointFormatVer"
      "sion\"5\n\027CheckpointFormatVersion\022\n\n\006LEGAC"
      "Y\020\000\022\006\n\002V1\020\001\022\006\n\002V2\020\002Be\n\023org.tensorflow.ut"
      "ilB\013SaverProtosP\001Z<github.com/tensorflow"
      "/tensorflow/tensorflow/go/core/protobuf\370"
      "\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 450);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensorflow/core/protobuf/saver.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto
namespace tensorflow {
const ::google::protobuf::EnumDescriptor* SaverDef_CheckpointFormatVersion_descriptor() {
  protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::file_level_enum_descriptors[0];
}
bool SaverDef_CheckpointFormatVersion_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const SaverDef_CheckpointFormatVersion SaverDef::LEGACY;
const SaverDef_CheckpointFormatVersion SaverDef::V1;
const SaverDef_CheckpointFormatVersion SaverDef::V2;
const SaverDef_CheckpointFormatVersion SaverDef::CheckpointFormatVersion_MIN;
const SaverDef_CheckpointFormatVersion SaverDef::CheckpointFormatVersion_MAX;
const int SaverDef::CheckpointFormatVersion_ARRAYSIZE;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

// ===================================================================

void SaverDef::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SaverDef::kFilenameTensorNameFieldNumber;
const int SaverDef::kSaveTensorNameFieldNumber;
const int SaverDef::kRestoreOpNameFieldNumber;
const int SaverDef::kMaxToKeepFieldNumber;
const int SaverDef::kShardedFieldNumber;
const int SaverDef::kKeepCheckpointEveryNHoursFieldNumber;
const int SaverDef::kVersionFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

SaverDef::SaverDef()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::scc_info_SaverDef.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.SaverDef)
}
SaverDef::SaverDef(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::scc_info_SaverDef.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.SaverDef)
}
SaverDef::SaverDef(const SaverDef& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  filename_tensor_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.filename_tensor_name().size() > 0) {
    filename_tensor_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.filename_tensor_name(),
      GetArenaNoVirtual());
  }
  save_tensor_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.save_tensor_name().size() > 0) {
    save_tensor_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.save_tensor_name(),
      GetArenaNoVirtual());
  }
  restore_op_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.restore_op_name().size() > 0) {
    restore_op_name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.restore_op_name(),
      GetArenaNoVirtual());
  }
  ::memcpy(&max_to_keep_, &from.max_to_keep_,
    static_cast<size_t>(reinterpret_cast<char*>(&version_) -
    reinterpret_cast<char*>(&max_to_keep_)) + sizeof(version_));
  // @@protoc_insertion_point(copy_constructor:tensorflow.SaverDef)
}

void SaverDef::SharedCtor() {
  filename_tensor_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  save_tensor_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  restore_op_name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&max_to_keep_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&version_) -
      reinterpret_cast<char*>(&max_to_keep_)) + sizeof(version_));
}

SaverDef::~SaverDef() {
  // @@protoc_insertion_point(destructor:tensorflow.SaverDef)
  SharedDtor();
}

void SaverDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  filename_tensor_name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  save_tensor_name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  restore_op_name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void SaverDef::ArenaDtor(void* object) {
  SaverDef* _this = reinterpret_cast< SaverDef* >(object);
  (void)_this;
}
void SaverDef::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void SaverDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* SaverDef::descriptor() {
  ::protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const SaverDef& SaverDef::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::scc_info_SaverDef.base);
  return *internal_default_instance();
}


void SaverDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.SaverDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  filename_tensor_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  save_tensor_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  restore_op_name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  ::memset(&max_to_keep_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&version_) -
      reinterpret_cast<char*>(&max_to_keep_)) + sizeof(version_));
  _internal_metadata_.Clear();
}

bool SaverDef::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.SaverDef)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string filename_tensor_name = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_filename_tensor_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->filename_tensor_name().data(), static_cast<int>(this->filename_tensor_name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "tensorflow.SaverDef.filename_tensor_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string save_tensor_name = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_save_tensor_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->save_tensor_name().data(), static_cast<int>(this->save_tensor_name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "tensorflow.SaverDef.save_tensor_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string restore_op_name = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(26u /* 26 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_restore_op_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->restore_op_name().data(), static_cast<int>(this->restore_op_name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "tensorflow.SaverDef.restore_op_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 max_to_keep = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(32u /* 32 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &max_to_keep_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool sharded = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(40u /* 40 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &sharded_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // float keep_checkpoint_every_n_hours = 6;
      case 6: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(53u /* 53 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &keep_checkpoint_every_n_hours_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
      case 7: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(56u /* 56 & 0xFF */)) {
          int value;
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   int, ::google::protobuf::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
          set_version(static_cast< ::tensorflow::SaverDef_CheckpointFormatVersion >(value));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.SaverDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.SaverDef)
  return false;
#undef DO_
}

void SaverDef::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.SaverDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string filename_tensor_name = 1;
  if (this->filename_tensor_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->filename_tensor_name().data(), static_cast<int>(this->filename_tensor_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.filename_tensor_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->filename_tensor_name(), output);
  }

  // string save_tensor_name = 2;
  if (this->save_tensor_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->save_tensor_name().data(), static_cast<int>(this->save_tensor_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.save_tensor_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->save_tensor_name(), output);
  }

  // string restore_op_name = 3;
  if (this->restore_op_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->restore_op_name().data(), static_cast<int>(this->restore_op_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.restore_op_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      3, this->restore_op_name(), output);
  }

  // int32 max_to_keep = 4;
  if (this->max_to_keep() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(4, this->max_to_keep(), output);
  }

  // bool sharded = 5;
  if (this->sharded() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(5, this->sharded(), output);
  }

  // float keep_checkpoint_every_n_hours = 6;
  if (this->keep_checkpoint_every_n_hours() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(6, this->keep_checkpoint_every_n_hours(), output);
  }

  // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
  if (this->version() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteEnum(
      7, this->version(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.SaverDef)
}

::google::protobuf::uint8* SaverDef::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SaverDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string filename_tensor_name = 1;
  if (this->filename_tensor_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->filename_tensor_name().data(), static_cast<int>(this->filename_tensor_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.filename_tensor_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->filename_tensor_name(), target);
  }

  // string save_tensor_name = 2;
  if (this->save_tensor_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->save_tensor_name().data(), static_cast<int>(this->save_tensor_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.save_tensor_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->save_tensor_name(), target);
  }

  // string restore_op_name = 3;
  if (this->restore_op_name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->restore_op_name().data(), static_cast<int>(this->restore_op_name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.restore_op_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        3, this->restore_op_name(), target);
  }

  // int32 max_to_keep = 4;
  if (this->max_to_keep() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(4, this->max_to_keep(), target);
  }

  // bool sharded = 5;
  if (this->sharded() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(5, this->sharded(), target);
  }

  // float keep_checkpoint_every_n_hours = 6;
  if (this->keep_checkpoint_every_n_hours() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(6, this->keep_checkpoint_every_n_hours(), target);
  }

  // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
  if (this->version() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteEnumToArray(
      7, this->version(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SaverDef)
  return target;
}

size_t SaverDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.SaverDef)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string filename_tensor_name = 1;
  if (this->filename_tensor_name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->filename_tensor_name());
  }

  // string save_tensor_name = 2;
  if (this->save_tensor_name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->save_tensor_name());
  }

  // string restore_op_name = 3;
  if (this->restore_op_name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->restore_op_name());
  }

  // int32 max_to_keep = 4;
  if (this->max_to_keep() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->max_to_keep());
  }

  // bool sharded = 5;
  if (this->sharded() != 0) {
    total_size += 1 + 1;
  }

  // float keep_checkpoint_every_n_hours = 6;
  if (this->keep_checkpoint_every_n_hours() != 0) {
    total_size += 1 + 4;
  }

  // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
  if (this->version() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::EnumSize(this->version());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SaverDef::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.SaverDef)
  GOOGLE_DCHECK_NE(&from, this);
  const SaverDef* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const SaverDef>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.SaverDef)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.SaverDef)
    MergeFrom(*source);
  }
}

void SaverDef::MergeFrom(const SaverDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.SaverDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.filename_tensor_name().size() > 0) {
    set_filename_tensor_name(from.filename_tensor_name());
  }
  if (from.save_tensor_name().size() > 0) {
    set_save_tensor_name(from.save_tensor_name());
  }
  if (from.restore_op_name().size() > 0) {
    set_restore_op_name(from.restore_op_name());
  }
  if (from.max_to_keep() != 0) {
    set_max_to_keep(from.max_to_keep());
  }
  if (from.sharded() != 0) {
    set_sharded(from.sharded());
  }
  if (from.keep_checkpoint_every_n_hours() != 0) {
    set_keep_checkpoint_every_n_hours(from.keep_checkpoint_every_n_hours());
  }
  if (from.version() != 0) {
    set_version(from.version());
  }
}

void SaverDef::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.SaverDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SaverDef::CopyFrom(const SaverDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.SaverDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SaverDef::IsInitialized() const {
  return true;
}

void SaverDef::Swap(SaverDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    SaverDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void SaverDef::UnsafeArenaSwap(SaverDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void SaverDef::InternalSwap(SaverDef* other) {
  using std::swap;
  filename_tensor_name_.Swap(&other->filename_tensor_name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  save_tensor_name_.Swap(&other->save_tensor_name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  restore_op_name_.Swap(&other->restore_op_name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(max_to_keep_, other->max_to_keep_);
  swap(sharded_, other->sharded_);
  swap(keep_checkpoint_every_n_hours_, other->keep_checkpoint_every_n_hours_);
  swap(version_, other->version_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata SaverDef::GetMetadata() const {
  protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::tensorflow::SaverDef* Arena::CreateMaybeMessage< ::tensorflow::SaverDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::SaverDef >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
