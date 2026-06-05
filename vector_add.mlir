module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32) {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<128x!tt.ptr<f32, 1>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32, 1>>, tensor<128xi32>
    %9 = tt.load %8, %6 : tensor<128x!tt.ptr<f32, 1>>
    %10 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<128x!tt.ptr<f32, 1>>
    %11 = tt.addptr %10, %4 : tensor<128x!tt.ptr<f32, 1>>, tensor<128xi32>
    %12 = tt.load %11, %6 : tensor<128x!tt.ptr<f32, 1>>
    %13 = arith.addf %9, %12 : tensor<128xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<128x!tt.ptr<f32, 1>>
    %15 = tt.addptr %14, %4 : tensor<128x!tt.ptr<f32, 1>>, tensor<128xi32>
    tt.store %15, %13, %6 : tensor<128x!tt.ptr<f32, 1>>
    tt.return
  }
}
