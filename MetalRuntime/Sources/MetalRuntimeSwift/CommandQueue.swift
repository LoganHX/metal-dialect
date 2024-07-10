import Metal
import MetalPerformanceShaders


@objc
public class CommandQueue: Wrappable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
    }
    
    @objc
    public func makeCommandBuffer(libPath: String,
                                  functionName: String,
                                  width: Int,
                                  height: Int,
                                  depth: Int) -> CommandBuffer? {
        do {
            let library = try device.makeLibrary(filepath: libPath)
            guard let function = library.makeFunction(name: functionName),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                return nil
            }
            let pipelineState = try device.makeComputePipelineState(function: function)
            computeEncoder.setComputePipelineState(pipelineState)
            
            let gridSize = MTLSize(width: width, height: height, depth: depth)
            var w = pipelineState.threadExecutionWidth
            w = w > gridSize.width ? gridSize.width : w
            var h = pipelineState.maxTotalThreadsPerThreadgroup / w
            h = h > gridSize.height ? gridSize.height : h
            let threads = MTLSize(width: w, height: h, depth: 1);
            
            return CommandBuffer(device: device,
                                 commandBuffer: commandBuffer,
                                 computeEncoder: computeEncoder,
                                 gridSize: gridSize,
                                 threads: threads)
        } catch {
            print(error)
            return nil
        }
    }
    
    
    @objc
    public func matMul(matA:UnsafeMutableRawPointer, rowsA: Int, columnsA: Int,
                       matB:UnsafeMutableRawPointer, rowsB: Int, columnsB: Int,
                       matC:UnsafeMutableRawPointer, elementSize: Int, typeName: String,
                       transposeA: Bool, transposeB: Bool) -> CommandBuffer? {
        
        let dataType: MPSDataType = getDataType(typeName: typeName)
        guard dataType != .invalid else {
            return nil
        }
        
        if(transposeA && transposeB){
            return nil;
        }
        
        let bufferA = device.makeBuffer(bytesNoCopy: matA,
                                        length: calculateAlignmentSize(size: rowsA*columnsA*elementSize),
                                        options: .storageModeShared)
        let bufferB = device.makeBuffer(bytesNoCopy: matB,
                                        length: calculateAlignmentSize(size: rowsB*columnsB*elementSize),
                                        options: .storageModeShared)
        let bufferC = device.makeBuffer(bytesNoCopy: matC,
                                        length: calculateAlignmentSize(size: rowsA*columnsB*elementSize),
                                        options: .storageModeShared)
        
        let descriptorA = MPSMatrixDescriptor(rows: rowsA, 
                                              columns: columnsA,
                                              rowBytes: columnsA*elementSize,
                                              dataType: dataType)
        let descriptorB = MPSMatrixDescriptor(rows: rowsB, 
                                              columns: columnsB,
                                              rowBytes: columnsB*elementSize,
                                              dataType: dataType)
        let descriptorC = MPSMatrixDescriptor(rows: rowsA, 
                                              columns: columnsB,
                                              rowBytes: columnsB*elementSize,
                                              dataType: dataType)
        
        let matrixA = MPSMatrix(buffer: bufferA!, descriptor: descriptorA)
        let matrixB = MPSMatrix(buffer: bufferB!, descriptor: descriptorB)
        let matrixC = MPSMatrix(buffer: bufferC!, descriptor: descriptorC)
        
        
        let matrixMultiplication = MPSMatrixMultiplication(device: device,
                                                           transposeLeft: transposeA,
                                                           transposeRight: transposeB,
                                                           resultRows: rowsA,
                                                           resultColumns: columnsB,
                                                           interiorColumns: columnsA,
                                                           alpha: 1.0, beta: 0.0)
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else { return nil }
        
        matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
        return CommandBuffer(device: device, commandBuffer: commandBuffer)
        
    }
    @objc
    public func printMat(mat:UnsafeMutableRawPointer, rows: Int, columns: Int, elementSize: Int) {
        
        let bufferA = device.makeBuffer(bytes: mat,
                                        length: rows*columns*elementSize,
                                        options: .storageModeShared)
        
        
        let descriptorA = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns*elementSize, dataType: MPSDataType.float32)
        
        let matrixA = MPSMatrix(buffer: bufferA!, descriptor: descriptorA)
        
        let buffer = matrixA.data
        let rowCount = matrixA.rows
        let columnCount = matrixA.columns
        
        
        // Otteniamo un puntatore ai dati nel buffer
        let pointer = buffer.contents()
        
        // Iteriamo attraverso ogni riga e colonna della matrice
        for row in 0..<rowCount {
            for col in 0..<columnCount {
                //(z * xSize * ySize) + (y * xSize) + x;
                
                // Calcoliamo l'indice nel buffer per l'elemento corrente
                let index = col + row * columnCount
                // Leggiamo il valore float dal buffer
                let value = pointer.load(fromByteOffset: index * elementSize, as: Float.self)
                // Stampiamo il valore
                print("\(value)", terminator: "\t")
            }
            print("") // Vai a capo dopo ogni riga
        }
        
        return;
    }
    
}
