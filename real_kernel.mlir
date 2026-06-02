module {
  tt.func public @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    
    // Here are the lines your encoder knows how to handle!
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %7 = tt.load %6 {cache = 1 : i32} : tensor<1024xf32>
    
    tt.return
  }
}