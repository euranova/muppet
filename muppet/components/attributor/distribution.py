"""Probability distribution-based attributors for MUPPET XAI.

This module provides attribution methods that analyze changes in probability distributions
over time, particularly designed for time series and sequential data explanation. These
attributors measure how perturbations affect the model's distributional predictions and
temporal dynamics.

Classes:
    ProbaShiftAttributor: Calculates attributions based on the difference between
        temporal distribution shifts and perturbation-induced distribution changes,
        implementing the FIT (Feature Importance in Time) methodology for time series
        explanation.
"""

#
# Created on Tue Apr 18 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch

from muppet import logger
from muppet.components.attributor.base import Attributor
from muppet.components.memory.base import Memory, PremiseList


class ProbaShiftAttributor(Attributor):
    """Attribution based on probability distribution shifts for time series classification.

    The distribution-based attributors are especially valuable for understanding sequential
    models where the temporal evolution of predictions is as important as the final output.
    They quantify feature importance by analyzing distributional shifts caused by perturbations.

    This attributor works on probability distributions over classes for classification tasks.
    The attribution is calculated as the difference between $KL(P(y|X_{0:t}) || P(y|X_{0:t-1}))$
    and $KL(P(y|X_{0:t}) || P(y|X'_{0:t}))$ summed over all classes, where $X'_{0:t}$ means the
    values of features are perturbed at time t.

    This attributor calculates feature importance based on probability distribution shifts
    over temporal sequences, implementing the FIT methodology for time series explanation.

    Attributes:
        outputs: Key-value mapping of timestep to model's output when calculating P(y|X_0:t).
        padding: The padding strategy for time series inputs.
        convention: The attribution convention (destructive).

    """

    def __init__(self, padding: str) -> None:
        """Initialize the ProbaShiftAttributor.

        Args:
            padding: Padding strategy for sequences. Options:
                - "left": Zero-pad sequences on the left (common for RNNs)
                - "right": Zero-pad sequences on the right
                - None: No padding for models handling variable lengths
        """
        self.padding = padding
        self.outputs = {}  # key: t, value: p(y/ x0:t)
        self.convention = "destructive"
        # The attribution computed as the difference explained in the doc string is high
        # when the perturbation impacts the model score on the selected class.
        super().__init__()

    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: Memory = PremiseList,
    ) -> None:
        """For every premise stored in the memory, fills up its attribution calculated from

        $$\sum_{\text{over all classes}}KL(P(y/X_{0:t}) || P(y/X_{0:t-1})) - \sum_{\text{over all classes}} KL(P(y/X_{0:t}) || P(y/X'_{0:t}))$$

        Args:
            x (torch.Tensor): The input example to be explained. Shape (b=1, f, t)
            perturbed_inputs (torch.Tensor): The calculated perturbations by the Perturbator. Shape (N, *x.shape)
            model (torch.nn.Module): The black-box model.
            memory (Memory, optional): The simple list memory structure.

        """
        idx_groups = {}
        premises_groups = {}

        for idx, premise in enumerate(memory.get_premises()):
            group_key = premise.key[0]["timestep"], premise.key[0]["feature"]
            idx_groups[group_key] = idx_groups.get(group_key, []) + [idx]
            premises_groups[group_key] = premises_groups.get(group_key, []) + [
                premise
            ]

        for ((time_step, feature), indexes), (_, premises) in zip(
            idx_groups.items(), premises_groups.items()
        ):
            # get original x0:t
            xt = x[:, :, : time_step + 1]  # (b, f, t)
            # get original X0:t-1
            xt_1 = xt[:, :, :-1]
            # get perturbed x at t (0:t-1 values of x, at t:=sampled value, then t+1:end Nans ignore them)
            xt_hat = perturbed_inputs[
                indexes, :, :, : time_step + 1
            ]  # (N, b, f, t)

            # if padding is provided ==> pad
            if self.padding:
                # number of values to pad for xt and xt_hat
                num_pad_values = abs(time_step + 1 - x.shape[2])

                xt = self._padd(x=xt, num_pad_values=num_pad_values, value=0)
                xt_1 = self._padd(
                    x=xt_1, num_pad_values=num_pad_values + 1, value=0
                )
                xt_hat = self._padd(
                    x=xt_hat, num_pad_values=num_pad_values, value=0
                )

            with torch.no_grad():
                try:
                    model_output = model(xt)
                except ValueError as e:
                    if "This classifier cannot handle unequal length" in str(
                        e
                    ):  # AEON model exception
                        logger.exception(
                            "The model cannot handle unequal length, consider using a padding strategy. "
                            "Not that FIT explainer might ont be the most appropriate for these models. "
                        )
                    raise e
                kl_epsilon = 1e-6
                output_xt = self.outputs.get(
                    time_step, model_output.detach()
                )  # p(y/ x0:t) (b, c) c:number of output classes
                output_xt_1 = self.outputs.get(
                    time_step - 1, model(xt_1).detach()
                )  # p(y/ x0:t-1) (b, c)
                output_xt_log = torch.log(output_xt + kl_epsilon)
                output_xt_1_log = torch.log(output_xt_1 + kl_epsilon)
                # mean here for the monte carlo estimation of $p(y|X_{O..t-1}, X_{S,t})$
                try:
                    output_xt_hat = model(xt_hat)
                except TypeError:
                    logger.debug(
                        "The prediction model do not handle batched data"
                    )
                    output_xt_hat = torch.stack(list(map(model, xt_hat)))

                output_xt_hat_log = torch.log(
                    output_xt_hat + kl_epsilon
                )  # p(y/ x'0:t)

            # fill the attributions dicts if not already done
            if time_step not in self.outputs.keys():
                self.outputs[time_step] = output_xt

            if (time_step - 1) not in self.outputs.keys():
                self.outputs[time_step - 1] = output_xt_1

            # TODO in terms of logic these KL computations should be moved to the proba aggregator (which should also be renamed).
            # calculate KL(p(xt)||p(y/xt-1)) as term1
            temporal_distribution_shift = torch.nn.KLDivLoss(
                reduction="batchmean", log_target=True
            )(output_xt_1_log, output_xt_log)

            # calculate KL(p(xt)||p(y/x't)) as term2
            unexplained_distribution_shift = torch.nn.KLDivLoss(
                reduction="batchmean", log_target=True
            )(output_xt_hat_log, output_xt_log)

            # fill up the premise's attribution
            FIT_importance_score_for_S = (
                temporal_distribution_shift - unexplained_distribution_shift
            )  # (b)

            # FIT importance score represent the hability of the complementary of
            # the perturbed feature to explain the temporal shift of the prediction
            # by taking the opposite we obtain the importance score for the pertubed feature.
            # FIT score > 0 the the complementary of the {perturbed feature} explains the temporal shift
            # FIT score = 0 the {all_features} - {perturbed feature} explains the temporal shift
            # FIT score < 0 the all_features - {perturbed feature} explains the temporal shift
            for premise in premises:
                # Apply the distributionnal aggregated score to all premise for the given (feature, timestep) couple
                premise.attribution = -FIT_importance_score_for_S

        return

    def _padd(self, x, num_pad_values, value):
        # left impute with 0
        if self.padding == "left":
            pad = (num_pad_values, 0)

        # right impute with 0
        if self.padding == "right":
            pad = (0, num_pad_values)

        return torch.nn.functional.pad(x, pad=pad, mode="constant", value=value)
