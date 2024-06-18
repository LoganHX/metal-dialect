public func calculateAlignmentSize(size: Int, alignment: Int = 4096) -> Int {
    let alignmentRemainder = size % alignment
    return alignmentRemainder == 0 ? size : size + (alignment - alignmentRemainder)
}
