import Foundation
import FluidAudio
import AVFoundation

// MARK: - Command Line Arguments
struct Arguments {
    let audioPath: String
    let outputPath: String

    init?(args: [String]) {
        guard args.count >= 3 else {
            print("Usage: swift vad_fluid.swift <audio_path> <output_path>")
            return nil
        }

        self.audioPath = args[1]
        self.outputPath = args[2]
    }
}

// MARK: - Audio Loading Utilities
func loadAudioFile(path: String) async throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let audioFile = try AVAudioFile(forReading: url)

    // Convert to 16kHz mono as required by FluidAudio
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    let converter = AVAudioConverter(from: audioFile.processingFormat, to: format)!

    let frameCount = AVAudioFrameCount(audioFile.length)
    let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!

    var error: NSError?
    let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
        let inputBuffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: inNumPackets)!
        do {
            try audioFile.read(into: inputBuffer)
            outStatus.pointee = .haveData
            return inputBuffer
        } catch {
            outStatus.pointee = .noDataNow
            return nil
        }
    }

    converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)

    guard error == nil else {
        throw error!
    }

    let samples = Array(UnsafeBufferPointer(start: outputBuffer.floatChannelData![0], count: Int(outputBuffer.frameLength)))
    return samples
}

// MARK: - VAD Output Structure
struct VADResult: Codable {
    let processedAt: String
    let method: String
    let audioDuration: Float
    let segments: [VADSegment]
    let metadata: VADMetadata
}

struct VADSegment: Codable {
    let start: Float
    let end: Float
    let duration: Float
}

struct VADMetadata: Codable {
    let device: String
    let processingTimeSeconds: Float
    let segmentCount: Int
}

// MARK: - Main Execution
@main
struct FluidVAD {
    static func main() async {
        let commandLineArgs = CommandLine.arguments

        guard let args = Arguments(args: commandLineArgs) else {
            exit(1)
        }

        do {
            let startTime = Date()

            print("Initializing FluidAudio VAD...")
            let vadManager = try await VadManager()

            print("Loading audio file: \(args.audioPath)")
            let audioSamples = try await loadAudioFile(path: args.audioPath)
            let audioDuration = Float(audioSamples.count) / 16000.0 // 16kHz sample rate

            print("Running VAD on \(audioSamples.count) samples (\(audioDuration)s)")

            // Use segmentSpeech to get speech segments directly
            let config = VadSegmentationConfig(
                minSpeechDuration: 0.0,
                minSilenceDuration: 0.0,
                speechPadding: 0.0
            )
            let speechSegments = try await vadManager.segmentSpeech(audioSamples, config: config)

            let processingTime = Float(Date().timeIntervalSince(startTime))
            print("VAD completed in \(processingTime)s")
            print("Found \(speechSegments.count) speech segments")

            // Convert to output format
            let segments = speechSegments.map { segment in
                VADSegment(
                    start: Float(segment.startTime),
                    end: Float(segment.endTime),
                    duration: Float(segment.endTime - segment.startTime)
                )
            }

            let vadResult = VADResult(
                processedAt: ISO8601DateFormatter().string(from: Date()),
                method: "fluid_audio_vad",
                audioDuration: audioDuration,
                segments: segments,
                metadata: VADMetadata(
                    device: "mps",
                    processingTimeSeconds: processingTime,
                    segmentCount: segments.count
                )
            )

            // Write result to JSON file
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(vadResult)

            try jsonData.write(to: URL(fileURLWithPath: args.outputPath))

            print("Results written to: \(args.outputPath)")

        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }
}
