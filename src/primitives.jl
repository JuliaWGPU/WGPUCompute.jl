
using WGPU
using GPUArrays

WGPU.SetLogLevel(WGPU.WGPULogLevel_Off)

gpuDevice = WGPU.getDefaultDevice();

(buffer, _) = WGPU.createBufferWithData(
	gpuDevice,
	"",
	rand(100, 100, 3),
	["Storage", "MapWrite"]
);

struct WGPUBuffer
	gpuDevice
	array
end

