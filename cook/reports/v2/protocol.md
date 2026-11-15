# Version 2 evaluation protocol

The version 1 model remains frozen. Version 2 uses four previously observed external matches for training (J03WPY, J03WN1, J03WMX, J03WOH), one new development match (J03WQQ), and two previously unscored external test matches (J03WOY, J03WR9).

The development search compares three classifiers (15 leaves, 31 leaves, and 31 leaves with short-spell weighting), three mixture weights and four sequence-evidence scales: 36 configurations. The selected configuration is frozen before either test match is scored. The public competition sample is never read by this experiment.

Version 2 is recommended only if its spellF1 exceeds frozen version 1 on both new test matches. No further settings will be chosen from these test results. Test scores are not used to refit any weights. If the candidate passes this check, one post-selection public sanity check may verify pipeline correctness; it will not change the model choice.

Version 1's original external test match is part of version 2 training, so that earlier result cannot be used as independent evidence about version 2.
