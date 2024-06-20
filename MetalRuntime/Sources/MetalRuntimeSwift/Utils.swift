import MetalPerformanceShaders

public func calculateAlignmentSize(size: Int, alignment: Int = 4096) -> Int {
    let alignmentRemainder = size % alignment
    return alignmentRemainder == 0 ? size : size + (alignment - alignmentRemainder)
}

public func getDataType(typeName: String) -> MPSDataType {
    let dataType: MPSDataType
    if(typeName == "float"){
        dataType = MPSDataType.float32;
    }
    else if(typeName == "int16_t"){
        dataType = MPSDataType.int16;
    }
    else if(typeName == "int8_t"){
        dataType = MPSDataType.int8;
    }
    else {
        dataType = MPSDataType.invalid;
    }
    
    return dataType;
}

