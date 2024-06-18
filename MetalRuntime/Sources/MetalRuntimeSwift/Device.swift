import Metal

@objc
public class Device: Wrappable {
  private let device: MTLDevice
  
  @objc
  public static func makeDefault() -> Device? {
    guard let device = MTLCreateSystemDefaultDevice() else {
      return nil
    }
    return Device(device: device)
  }
  
  private init(device: MTLDevice) {
    self.device = device
  }
  
  @objc
  public func makeCommandQueue() -> CommandQueue? {
    guard let commandQueue = device.makeCommandQueue() else {
      return nil
    }
    return CommandQueue(device: device, commandQueue: commandQueue)
  }
  
  
    @objc public func makeBuffer(isStorageModeManaged: Bool, bufferSize: Int, count: Int) -> Buffer? {
      let option: MTLResourceOptions = isStorageModeManaged
        ? .storageModeManaged
        : .storageModeShared
        var alignm: Int
        
        alignm = calculateAlignmentSize(size: bufferSize)
        
      if let buffer = device.makeBuffer(length: alignm, options: option) {
        return Buffer(buffer: buffer, count: count)
      }
      return nil
    }

    
}
