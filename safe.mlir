module {
  tt.func public @kernel(%arg0: !tt.ptr<f32, 1>, %arg1: i32) {
    %c128_i32 = "arith.constant"() {value = 128 : i32} : () -> i32
    %0 = "tt.get_program_id"() {axis = 0 : i32} : () -> i32
    %1 = "arith.muli"(%0, %c128_i32) : (i32, i32) -> i32
    %2 = "tt.make_range"() {end = 128 : i32, start = 0 : i32} : () -> tensor<128xi32>
    %3 = "tt.splat"(%1) : (i32) -> tensor<128xi32>
    %4 = "arith.addi"(%3, %2) : (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
    %5 = "tt.splat"(%arg0) : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>>
    %6 = "tt.addptr"(%5, %4) : (tensor<128x!tt.ptr<f32, 1>>, tensor<128xi32>) -> tensor<128x!tt.ptr<f32, 1>>
    %7 = "tt.load"(%6) : (tensor<128x!tt.ptr<f32, 1>>) -> tensor<128xf32>
    tt.return
  }
}