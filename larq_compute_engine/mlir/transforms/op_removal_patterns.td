include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Partially supported in TFLite, treated as passthrough IdentityOp
def : Pat<(TF_CheckNumericsOp $arg, $msg), (TF_IdentityOp $arg)>;
def : Pat<(TF_SnapshotOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_StopGradientOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_PlaceholderWithDefaultOp $arg), (TF_IdentityOp $arg)>;

//===----------------------------------------------------------------------===//
// Op removal patterns.
//===----------------------------------------------------------------------===//
def : Pat<(TF_IdentityOp $arg), (replaceWithValue $arg)>;
def : Pat<(TF_IdentityNOp $arg), (replaceWithValue $arg)>;
