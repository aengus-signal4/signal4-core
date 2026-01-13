import Foundation
import FluidAudio
import AVFoundation

// MARK: - Command Line Arguments
struct Arguments {
    let audioPath: String
    let outputPath: String
    let threshold: Float
    
    init?(args: [String]) {
        guard args.count >= 3 else {
            print("Usage: swift diarize_fluid.swift <audio_path> <output_path> [threshold=0.7]")
            return nil
        }
        
        self.audioPath = args[1]
        self.outputPath = args[2]
        self.threshold = args.count > 3 ? Float(args[3]) ?? 0.7 : 0.7
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

// MARK: - Diarization Output Structure
struct DiarizationResult: Codable {
    let contentId: String
    let processedAt: String
    let method: String
    let model: String
    let audioDuration: Float
    let speakersDetected: Int
    let segments: [DiarizationSegment]
    let speakerEmbeddings: [String: [Float]]?
    let metadata: DiarizationMetadata
}

struct DiarizationSegment: Codable {
    let start: Float
    let end: Float
    let duration: Float
    let speaker: String
}

struct DiarizationMetadata: Codable {
    let device: String
    let processingTimeSeconds: Float
    let threshold: Float
}

// MARK: - Main Execution
@main
struct FluidDiarizer {
    static func main() async {
        let commandLineArgs = CommandLine.arguments
        
        guard let args = Arguments(args: commandLineArgs) else {
            exit(1)
        }
        
        do {
            let startTime = Date()
            
            print("Loading FluidAudio models...")
            let models = try await DiarizerModels.downloadIfNeeded()
            
            print("Initializing diarizer with threshold: \(args.threshold)")
            let config = DiarizerConfig(clusteringThreshold: args.threshold, debugMode: true)
            let diarizer = DiarizerManager(config: config)
            diarizer.initialize(models: models)
            
            print("Loading audio file: \(args.audioPath)")
            let audioSamples = try await loadAudioFile(path: args.audioPath)
            let audioDuration = Float(audioSamples.count) / 16000.0 // 16kHz sample rate
            
            print("Performing diarization on \(audioSamples.count) samples (\(audioDuration)s)")
            let result = try await diarizer.performCompleteDiarization(audioSamples)
            
            let processingTime = Float(Date().timeIntervalSince(startTime))
            print("Diarization completed in \(processingTime)s")
            
            // Convert FluidAudio result to our format (backward compatible with stitch pipeline)
            let segments = result.segments.map { segment in
                DiarizationSegment(
                    start: segment.startTimeSeconds,
                    end: segment.endTimeSeconds,
                    duration: segment.endTimeSeconds - segment.startTimeSeconds,
                    speaker: "SPEAKER_\(segment.speakerId)"
                )
            }
            
            let uniqueSpeakers = Set(result.segments.map { $0.speakerId }).count
            
            let diarizationResult = DiarizationResult(
                contentId: URL(fileURLWithPath: args.audioPath).deletingPathExtension().lastPathComponent,
                processedAt: ISO8601DateFormatter().string(from: Date()),
                method: "fluid_audio",
                model: "FluidInference/speaker-diarization-coreml",
                audioDuration: audioDuration,
                speakersDetected: uniqueSpeakers,
                segments: segments,
                speakerEmbeddings: result.speakerDatabase,
                metadata: DiarizationMetadata(
                    device: "mps", // Assuming Apple Silicon Mac
                    processingTimeSeconds: processingTime,
                    threshold: args.threshold
                )
            )
            
            // Write result to JSON file
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(diarizationResult)
            
            try jsonData.write(to: URL(fileURLWithPath: args.outputPath))
            
            print("Results written to: \(args.outputPath)")
            print("Found \(uniqueSpeakers) speakers in \(segments.count) segments")
            
        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }
}