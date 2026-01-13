// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FluidDiarization",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.6.1"),
    ],
    targets: [
        .executableTarget(
            name: "FluidDiarization",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            path: ".",
            sources: ["diarize_fluid.swift"]
        ),
        .executableTarget(
            name: "FluidVAD",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            path: ".",
            sources: ["vad_fluid.swift"]
        )
    ]
)