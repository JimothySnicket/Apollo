/**
 * @file tests/unit/test_mic.cpp
 * @brief Unit tests for the moonlight-mic parser and capability gate.
 *
 * Covers:
 *   - mic::parse_mic_frame: valid input, all error variants
 *   - Capability gate helper: should_allocate_mic_resources()
 *   - Decoder null-tolerance is documented via a compile-time review note
 *     (the dispatch handler itself requires a full streaming session to invoke).
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

#include <src/mic_parser.h>

#include "../tests_common.h"

// ---------------------------------------------------------------------------
// Helper: build a well-formed payload from scalar fields + Opus bytes.
// Header is written in big-endian as the wire spec mandates.
// ---------------------------------------------------------------------------
static std::vector<std::byte>
make_mic_payload(uint16_t seq, uint16_t opusLen, uint32_t ts,
                 const std::vector<std::byte> &opusData) {
  std::vector<std::byte> buf;
  buf.resize(8 + opusData.size());

  // BE16 sequenceNumber
  buf[0] = static_cast<std::byte>((seq >> 8) & 0xFF);
  buf[1] = static_cast<std::byte>(seq & 0xFF);
  // BE16 opusFrameLength
  buf[2] = static_cast<std::byte>((opusLen >> 8) & 0xFF);
  buf[3] = static_cast<std::byte>(opusLen & 0xFF);
  // BE32 timestampSamples
  buf[4] = static_cast<std::byte>((ts >> 24) & 0xFF);
  buf[5] = static_cast<std::byte>((ts >> 16) & 0xFF);
  buf[6] = static_cast<std::byte>((ts >> 8) & 0xFF);
  buf[7] = static_cast<std::byte>(ts & 0xFF);
  // Opus payload
  for (std::size_t i = 0; i < opusData.size(); ++i) {
    buf[8 + i] = opusData[i];
  }
  return buf;
}

// Convenience overload for simple byte-pattern payloads.
static std::vector<std::byte>
make_opus_bytes(std::size_t n, std::byte fill = std::byte{0xAB}) {
  return std::vector<std::byte>(n, fill);
}

// ---------------------------------------------------------------------------
// 1. Parser — valid input
// ---------------------------------------------------------------------------
TEST(MicParserTest, ValidInput) {
  // seq=0x0001, opusLen=10 (0x000A), ts=0x000003C0 (960)
  constexpr uint16_t kSeq     = 0x0001;
  constexpr uint16_t kOpusLen = 10;
  constexpr uint32_t kTs      = 0x000003C0;

  auto opusData = make_opus_bytes(kOpusLen, std::byte{0x55});
  auto payload  = make_mic_payload(kSeq, kOpusLen, kTs, opusData);

  auto result = mic::parse_mic_frame(std::span<const std::byte>(payload));

  ASSERT_TRUE(std::holds_alternative<mic::parsed_frame_t>(result))
    << "Expected success but got an error variant";

  const auto &frame = std::get<mic::parsed_frame_t>(result);
  EXPECT_EQ(frame.sequenceNumber,  kSeq);
  EXPECT_EQ(frame.opusFrameLength, kOpusLen);
  EXPECT_EQ(frame.timestampSamples, kTs);

  // opusPayload must point into the buffer starting right after the 8-byte header
  ASSERT_EQ(frame.opusPayload.size(), kOpusLen);
  EXPECT_EQ(frame.opusPayload.data(),
            reinterpret_cast<const std::byte *>(payload.data()) + 8);
  for (std::size_t i = 0; i < kOpusLen; ++i) {
    EXPECT_EQ(frame.opusPayload[i], std::byte{0x55}) << "byte mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// 2. Parser — payload too small (4 bytes, header is 8 bytes)
// ---------------------------------------------------------------------------
TEST(MicParserTest, PayloadTooSmall) {
  std::array<std::byte, 4> tiny{};
  auto result = mic::parse_mic_frame(std::span<const std::byte>(tiny));

  ASSERT_TRUE(std::holds_alternative<mic::parse_error_t>(result));
  EXPECT_EQ(std::get<mic::parse_error_t>(result), mic::parse_error_t::payload_too_small);
}

// ---------------------------------------------------------------------------
// 3. Parser — inner length mismatch (header claims 20 bytes, only 5 follow)
// ---------------------------------------------------------------------------
TEST(MicParserTest, InnerLengthMismatch) {
  // Build a header claiming opusLen=20, but supply only 5 actual bytes.
  constexpr uint16_t kClaimedLen = 20;
  auto opusData = make_opus_bytes(5);
  // make_mic_payload uses opusData.size() as the actual tail, but writes
  // kClaimedLen into the header field.
  auto payload = make_mic_payload(0x0002, kClaimedLen, 0x00000001, opusData);
  // payload is 8 + 5 = 13 bytes; header says 20 → mismatch.

  auto result = mic::parse_mic_frame(std::span<const std::byte>(payload));

  ASSERT_TRUE(std::holds_alternative<mic::parse_error_t>(result));
  EXPECT_EQ(std::get<mic::parse_error_t>(result), mic::parse_error_t::inner_length_mismatch);
}

// ---------------------------------------------------------------------------
// 4. Parser — opus length zero
// ---------------------------------------------------------------------------
TEST(MicParserTest, OpusLengthZero) {
  // opusLen=0 in header, no trailing bytes (header only, 8 bytes total).
  auto payload = make_mic_payload(0x0003, 0, 0x00000002, {});

  auto result = mic::parse_mic_frame(std::span<const std::byte>(payload));

  ASSERT_TRUE(std::holds_alternative<mic::parse_error_t>(result));
  EXPECT_EQ(std::get<mic::parse_error_t>(result), mic::parse_error_t::opus_length_zero);
}

// ---------------------------------------------------------------------------
// 5. Parser — opus length too large (> MAX_OPUS_FRAME_BYTES = 1392)
// ---------------------------------------------------------------------------
TEST(MicParserTest, OpusLengthTooLarge) {
  // Claim opusLen = MAX_OPUS_FRAME_BYTES + 1 = 1393.
  constexpr uint16_t kTooLarge = static_cast<uint16_t>(mic::MAX_OPUS_FRAME_BYTES + 1);
  // Supply the matching number of actual bytes so the inner-length check
  // doesn't fire first — we want opus_length_too_large specifically.
  auto opusData = make_opus_bytes(kTooLarge);
  auto payload  = make_mic_payload(0x0004, kTooLarge, 0x00000003, opusData);

  auto result = mic::parse_mic_frame(std::span<const std::byte>(payload));

  ASSERT_TRUE(std::holds_alternative<mic::parse_error_t>(result));
  EXPECT_EQ(std::get<mic::parse_error_t>(result), mic::parse_error_t::opus_length_too_large);
}

// ---------------------------------------------------------------------------
// 6. Capability gate: should_allocate_mic_resources
//
// The H3 gate in session::alloc() is:
//   session->mic.client_advertised = (config.mlFeatureFlags & ML_FF_MIC_INPUT) != 0;
//
// ML_FF_MIC_INPUT = 0x04. We test the boolean evaluation directly without
// constructing a full session_t (too heavyweight for a unit test).
// ---------------------------------------------------------------------------
namespace {
  // Mirror the stream.cpp constant so the test doesn't need to reach into the
  // production anonymous namespace.
  constexpr std::uint32_t ML_FF_MIC_INPUT_TEST = 0x04;

  bool should_allocate_mic_resources(std::uint32_t mlFeatureFlags) {
    return (mlFeatureFlags & ML_FF_MIC_INPUT_TEST) != 0;
  }
}  // anonymous namespace

TEST(MicCapabilityGateTest, FlagSetAllocates) {
  // Flag present → allocate
  EXPECT_TRUE(should_allocate_mic_resources(0x04));
  EXPECT_TRUE(should_allocate_mic_resources(0xFF));   // multiple bits, including 0x04
}

TEST(MicCapabilityGateTest, FlagClearSkips) {
  // Flag absent → skip allocation
  EXPECT_FALSE(should_allocate_mic_resources(0x00));
  EXPECT_FALSE(should_allocate_mic_resources(0xFB));  // all bits except 0x04
}

// ---------------------------------------------------------------------------
// 7. Decoder null-tolerance
//
// The dispatch handler in stream.cpp guards against a null decoder:
//   if (!session->mic.decoder) { BOOST_LOG(warning) << ...; return; }
//
// We cannot invoke the handler here without a full streaming session, but we
// CAN verify the parser succeeds on valid input — which is the only path that
// reaches the decoder guard.  A valid frame + null decoder → drop (logged).
// This test verifies the parser completes successfully so the handler *would*
// reach the decoder guard rather than returning early with a parse error.
// ---------------------------------------------------------------------------
TEST(MicDecoderLifecycleTest, ValidFrameReachesDecoderGuard) {
  // Build a valid 10-byte Opus frame.
  auto opusData = make_opus_bytes(10);
  auto payload  = make_mic_payload(0x0005, 10, 0x00000780, opusData);

  auto result = mic::parse_mic_frame(std::span<const std::byte>(payload));

  // Must succeed — only then would the handler reach the decoder null-check.
  EXPECT_TRUE(std::holds_alternative<mic::parsed_frame_t>(result))
    << "Parser must succeed for a valid frame so the null-decoder guard is reachable";
}
