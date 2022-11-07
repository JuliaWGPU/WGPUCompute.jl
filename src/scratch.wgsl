
struct IOArray {
    data:array<f32>
};

@group(0) @binding(0) var<storage, read_write> input0:IOArray ;
@group(0) @binding(1) var<storage, read_write> ouput1:IOArray ;
@compute @workgroup_size(8, 8, 4) 
fn Relu(@builtin(global_invocation_id) global_id:vec3<u32>) { 
    let gIdx = global_id.x * global_id.y+global_id.z;
    let value = input0.data[gIdx];
    ouput1.data[gIdx] = max(value, 0.0);
}

