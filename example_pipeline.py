import sys

from ax3_pipeline.pipeline_blocks.compositions import SequentialComposition, ParallelComposition
from ax3_pipeline.pipeline_blocks.features import ApproximateVelocity, ApproximateDistance, TimeDomainSummaryStatistics, \
    EnmoSummaryStatistics, ActivityClasses, TotalEnergy, SpectralEntropy
from ax3_pipeline.pipeline_blocks.filters import LowpassButterworthFilter, HighpassButterworthFilter
from ax3_pipeline.pipeline_blocks.misc import EpochGenerator, TrapezoidalIntegrator, VectorMagnitude, \
    EuclideanNormMinusOne, Dft1d
from ax3_pipeline.pipeline_blocks.postprocessors import FeatureConcat, NonWeartimeCalculator


if __name__ == "__main__":

    timestamps_ms = sys.argv[1]
    x = sys.argv[2]
    y = sys.argv[3]
    z = sys.argv[4]

    epoch_gen = EpochGenerator(timestamps_ms=timestamps_ms)

    pipeline = SequentialComposition(
        LowpassButterworthFilter(),
        ParallelComposition(
            SequentialComposition(
                HighpassButterworthFilter(),
                epoch_gen,
                ParallelComposition(
                    SequentialComposition(
                        TrapezoidalIntegrator(),
                        VectorMagnitude(),
                        ParallelComposition(
                            ApproximateVelocity(),
                            SequentialComposition(
                                TrapezoidalIntegrator(cumulative=False),
                                ApproximateDistance()
                            )
                        )
                    ),
                    TimeDomainSummaryStatistics()
                )
            ),
            SequentialComposition(
                EuclideanNormMinusOne(),
                epoch_gen,
                ParallelComposition(
                    EnmoSummaryStatistics(),
                    ActivityClasses(),
                    SequentialComposition(
                        Dft1d(),
                        ParallelComposition(
                            TotalEnergy(),
                            SpectralEntropy()
                        )
                    )
                )
            )
        ),
        FeatureConcat(epoch_gen.timestamps),
        NonWeartimeCalculator()
    )

    df = pipeline.process(x, y, z)[0]
    df.to_csv("test_acc_pipeline.csv")
