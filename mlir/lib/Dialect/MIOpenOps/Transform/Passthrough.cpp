#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/MIOpenOps/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct Passthrough : public FunctionPass<Passthrough> {
  void runOnFunction() override;
};
} // anonymous namespace

void Passthrough::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](miopen::TransformOp op) {
    // dump the affine map from the source memref.
    auto inputType = op.input().getType().dyn_cast<MemRefType>();
    auto inputShape = inputType.getShape();
    auto inputRank = inputType.getRank();
    auto inputElementType = inputType.getElementType();
    auto inputAffineMaps = inputType.getAffineMaps();

    auto perm = AffineMap::getPermutationMap({2, 3, 0, 1}, op.getContext());
    perm.dump();

    //auto lm = makeStridedLinearLayoutMap({MemRefType::getDynamicStrideOrOffset(), MemRefType::getDynamicStrideOrOffset(), MemRefType::getDynamicStrideOrOffset(), 1}, 0, op.getContext());
    //lm.dump();

    //auto outputAffineMap = lm.compose(perm);
    //outputAffineMap.dump();

    //for (size_t i = 0; i < perm.getNumResults(); ++i) {
    //  perm.getResult(i).dump();
    //}
    //lm.getResult(0).dump();
    //outputAffineMap.getResult(0).dump();

    //auto diff = outputAffineMap.getResult(0) - lm.getResult(0);
    //diff.dump();

    auto outputType = op.output().getType().dyn_cast<MemRefType>();
    auto outputShape = outputType.getShape();
    auto outputElementType = outputType.getElementType();
    auto transformedOutputType = MemRefType::get(outputShape, outputType.getElementType(),
                                                 {perm});

    OpBuilder b(op.getOperation());
    auto loc = op.getLoc();
    auto newOp = b.create<miopen::TransformOp>(loc, transformedOutputType, op.input(), op.getAttrs());
    op.erase();
  });
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::miopen::createPassthroughPass() {
  return std::make_unique<Passthrough>();
}

static PassRegistration<Passthrough>
  pass("passthrough", "Handle passthrough miopen.transform");

