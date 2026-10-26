"""Constants of the pre-registered classification protocol (docs/CLASSIFICATION_PROTOCOL.md).

Values are copied unchanged from the classifier that produced the published classes.
"""

FAST_THRESHOLD_mS_cm = 0.1
T_REF_K = 300.0
D0_MIN_cm2_s = 1e-4
D0_MIN_SENSITIVITY_cm2_s = 1e-5
T_BIAS_MAX = 0.05
FRAMEWORK_MSD_MAX_A2 = 1.0
FRAMEWORK_FRACTION_MAX = 0.10
BOUND_LAG_PS = 100.0
UNSTABLE_LOWEST_T_K = 400.0
REPLICATE_MIN_SEEDS = 3
REPLICATE_MAX_RATIO = 2.0
REPLICATE_MIN_R2 = 0.95
TERMINAL_MAX_SEEDS = 5
TERMINAL_MAX_PRODUCTION_PS = 800.0
REPLICATE_PASS_DECISIONS = ("pass", "pass_seeds01", "pass_low_sigma")

FAST_VALIDATED = "FAST_VALIDATED"
SLOW_MEASURED = "SLOW_MEASURED"
SLOW_ESTIMATED = "SLOW_ESTIMATED"
SLOW_BOUNDED = "SLOW_BOUNDED"
THERMALLY_UNSTABLE = "THERMALLY_UNSTABLE"
NEEDS_COMPUTE = "NEEDS_COMPUTE"
TERMINAL_CLASSES = (FAST_VALIDATED, SLOW_MEASURED, SLOW_ESTIMATED, SLOW_BOUNDED, THERMALLY_UNSTABLE)
CLASSES = TERMINAL_CLASSES + (NEEDS_COMPUTE,)

# The compute list pre-registered in the protocol (batch 1), for comparison only.
PREREGISTERED_COMPUTE_LIST = ("Li20Si3P3S23Cl", "Li5P(S2Cl)2", "Li10Zn7(PS4)8", "Li3NbS4", "Li5BiS4",
                              "Li48P16S61", "Li9Nd2(PS4)5", "Li6AsS5I")

E_CHARGE_C = 1.602176634e-19
K_B_J_K = 1.380649e-23
K_B_EV_K = K_B_J_K / E_CHARGE_C

# Amendment A3.2: an eligible run whose kinisi D differs from its direct-MSD D by more
# than this factor (either way) is a failed fit.
ESTIMATOR_RATIO_FLAG = 3.0
