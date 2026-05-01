/**
 * @file src/mic_parser.h
 * @brief Pure parser for the moonlight-mic 0x5510 wire-format header.
 *
 * Wire format (docs/design/WIRE.md):
 *   [BE16 sequenceNumber][BE16 opusFrameLength][BE32 timestampSamples][<opusFrameLength bytes>]
 *
 * This header is included from both stream.cpp (production path) and the unit
 * tests.  It deliberately has NO dependencies on Boost.Log, stream session
 * state, or any platform-specific header — that is the point: pure function,
 * returns by value, zero side effects, trivially testable.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

#include <boost/endian/arithmetic.hpp>

namespace mic {

  // Maximum Opus payload bytes we will accept.
  // Derived from MIC_PACKET_MTU (1400) minus the 8-byte header.
  inline constexpr std::size_t MAX_OPUS_FRAME_BYTES = 1392;

  // The 8-byte big-endian header that precedes the Opus payload on the wire.
#pragma pack(push, 1)
  struct frame_header_t {
    boost::endian::big_uint16_at sequenceNumber;   // BE16; monotonic, wraps at 65535
    boost::endian::big_uint16_at opusFrameLength;  // BE16; byte length of the Opus payload
    boost::endian::big_uint32_at timestampSamples; // BE32; 48 kHz sample count since first frame
  };
#pragma pack(pop)

  static_assert(sizeof(frame_header_t) == 8, "frame_header_t must be 8 bytes");

  // Parsed output — all fields are native-endian copies; opusPayload is a
  // non-owning view into the caller-supplied buffer (caller must keep it alive).
  struct parsed_frame_t {
    uint16_t sequenceNumber;
    uint16_t opusFrameLength;
    uint32_t timestampSamples;
    std::span<const std::byte> opusPayload;
  };

  enum class parse_error_t {
    payload_too_small,       // total payload < 8 bytes (shorter than header)
    opus_length_zero,        // opusFrameLength == 0
    opus_length_too_large,   // opusFrameLength > MAX_OPUS_FRAME_BYTES
    inner_length_mismatch,   // header opusFrameLength != actual remaining bytes
  };

  /**
   * @brief Parse a raw 0x5510 payload into a validated frame descriptor.
   *
   * @param payload  Raw bytes exactly as received from the network (header + Opus data).
   * @return parsed_frame_t on success; parse_error_t on any validation failure.
   *
   * This function performs NO logging and has NO side effects. All
   * error handling (logging, session state mutation) is the caller's
   * responsibility.
   */
  inline std::variant<parsed_frame_t, parse_error_t>
  parse_mic_frame(std::span<const std::byte> payload) {
    if (payload.size() < sizeof(frame_header_t)) {
      return parse_error_t::payload_too_small;
    }

    const auto *hdr = reinterpret_cast<const frame_header_t *>(payload.data());
    const uint16_t seq      = static_cast<uint16_t>(hdr->sequenceNumber);
    const uint16_t opusLen  = static_cast<uint16_t>(hdr->opusFrameLength);
    const uint32_t tsSamples = static_cast<uint32_t>(hdr->timestampSamples);

    if (opusLen == 0) {
      return parse_error_t::opus_length_zero;
    }
    if (opusLen > MAX_OPUS_FRAME_BYTES) {
      return parse_error_t::opus_length_too_large;
    }
    if (sizeof(frame_header_t) + opusLen != payload.size()) {
      return parse_error_t::inner_length_mismatch;
    }

    auto opusPayload = payload.subspan(sizeof(frame_header_t), opusLen);

    return parsed_frame_t {
      .sequenceNumber  = seq,
      .opusFrameLength = opusLen,
      .timestampSamples = tsSamples,
      .opusPayload     = opusPayload,
    };
  }

}  // namespace mic
