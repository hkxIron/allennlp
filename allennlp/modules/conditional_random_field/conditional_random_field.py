"""
Conditional random field
"""
from typing import List, Tuple, Dict, Union

import torch

from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util

VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    # Parameters

    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    # Returns

    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    # Parameters

    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.

    # Returns

    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    # Parameters

    num_tags : `int`, required
        The number of tags.
    constraints : `List[Tuple[int, int]]`, optional (default = `None`)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default = `True`)
        Whether to include the start and end transition parameters.
    """

    def __init__(
        self,
        num_tags: int,
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 1.0)
        else:
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 0.0)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(
        self, logits: torch.Tensor, transitions: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        计算log P(y|x;w)中的分母，即:
         log p(s|x;w) = log[exp(w*feat(xi,j,s(j-1),s(j)))/Z], xi指第i条样本, s为state, j为第j个时间步, Z为归一化因子,
         分母为 logZ = log sum[exp(w*feat(xi,j,s(j-1),s(j)))], 其中feat(xi,j,s(j-1),s(j))包含了转移概率，发射概率

        Computes the (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood

        This is the sum of the likelihoods across all possible state sequences.

        Args:
            logits (torch.Tensor): a (batch_size, sequence_length num_tags) tensor of unnormalized log-probabilities
            transitions (torch.Tensor): a (num_tags, num_tags) tensor of transition scores
            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags

        Returns:
            torch.Tensor: (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        # mask:[batch,seq_len] -> [seq_len, batch]
        mask = mask.transpose(0, 1).contiguous()
        # logits:[batch,seq_len, num_tags] -> [seq_len, batch, num_tags]
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        # alpha(t, s):是前向后向算法里的前向节点,代表第t个时间步，到达状态s的概率
        if self.include_start_end_transitions:
            # start_transitions:[1,num_tags], 初始分布
            # logits[0]:[batch, num_tags]
            # alpha:[batch, num_tags]
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            # alpha:[batch, num_tags]
            alpha = logits[0]

        # alpha(t, s):是前向后向算法里的前向节点,代表第t个时间步，到达状态s的概率
        # psai(xi,st-1,st,t) = w*feat(xi, j, sj-1, sj), 这里包含了转移概率与发射概率
        # alpha(t,s) = sum_{s'} {alpha(t-1, s') * psai(s',s,t) }, s'为t-1时的状态
        # 在log下，所有的相乘均就为相加
        # For each t we compute logits for the transitions from timestep t-1 to timestep t.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for t in range(1, sequence_length): # [1, seq_len-1]
            # The emit scores are for time t ("next_tag") so we broadcast along the current_tag axis.
            # logits: [seq_len, batch, num_tags]
            # emit_scores:[batch, 1, num_tags]
            emit_scores = logits[t].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            # transitions:[num_tags, num_tags]
            # transition_scores:[1, num_tags, num_tags]
            transition_scores = transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            # alpha:[batch, num_tags, 1]
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            # inner:[batch, num_tags, num_tags], 即[样本下标,当前状态,下一状态]
            # 这里的inner就是 w*feat(xi,j,s(j-1),s(j)), 即转移分数 + 发射分数
            # alpha(t,s) = sum_{s'} {alpha(t-1, s') * psai(s',s,t) }, s'为t-1时的状态
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == True) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == False) we want to retain the previous alpha.
            # mask[t]:[batch]
            # inner:[batch, num_tags:prev_state, num_tags:cur_state],
            # tag_score:[batch, num_tags]
            tag_score = util.logsumexp(inner, 1) # dim=1是按当前cur_tag进行sum
            # alpha:[batch, num_tags]
            alpha = (tag_score * mask[t].view(batch_size, 1) + alpha * (~mask[t]).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        # stops:[batch, num_tags] -> [batch]
        return util.logsumexp(stops)

    def _joint_likelihood(
        self,
        logits: torch.Tensor,
        transitions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        计算log P(y|x;w)中的分子，即:
         log p(s|x;w) = log[exp(w*feat(xi,j,s(j-1),s(j)))/Z], xi指第i条样本, s为state, j为第j个时间步, Z为归一化因子,
         分子为w*feat(xi,j,s(j-1),s(j), 其中feat(xi,j,s(j-1),s(j))包含了转移概率，发射概率

         w*feat(xi,j,s(j-1),s(j))
         = sum_{k,j}[alpha_k*t_k(yj-1,yj,xi)] + sum_{l,j}*[beta_l* s_l(yj,xi,j)]
         = alpha*转移概率 + beta*发射概率
         其中t_k:代表第k个转移特征，s_l:代表第l个状态特征
         而在一般在DL中，t_k，s_l只有一个t,s，可以认为前面的NN已将这些相同时间步的不同特征相加得到一个总特征，
         即:t=sum_{k}{s_k}, s=sum_{l}{s_l}

        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        Args:
            logits (torch.Tensor): a (batch_size, sequence_length num_tags) tensor of unnormalized log-probabilities
            transitions (torch.Tensor): a (batch_size, num_tags, num_tags) tensor of transition scores
            tags (torch.Tensor): output tag sequences (batch_size, sequence_length) $y$ for each input sequence
            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags

        Returns:
            torch.Tensor: numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        # logits:[batch, seq_len, num_tags] -> [seq_len, batch, num_tags]
        # 在NN+CRF中，logits就是发射概率，直接取就行,而不是当成特征再乘以权重w
        logits = logits.transpose(0, 1).contiguous()
        # mask:[batch, seq_len] -> [seq_len, batch]
        mask = mask.transpose(0, 1).contiguous()
        # tags:[batch, seq_len] -> [seq_len, batch]
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            # start_transitions:[num_tags], 初始分布
            # tags:[seq_len, batch], tags[0]:[batch]
            # score:[batch]
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for t in range(sequence_length - 1): # 注意此处t的范围:0~len-2, 而应该包含的范围是：0~len-1
            # Each is shape (batch_size,)
            # current_tag:[batch], next_tag:[batch]
            current_tag, next_tag = tags[t], tags[t + 1]

            # The scores for transitioning from current_tag to next_tag, [batch]
            # transitions:[num_tags, num_tags]
            # transition_score:[batch] 获取tag[t]->tag[t+1]的转移分数, current_tag作为下标
            transition_score = transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            # logits: [seq_len, batch, num_tags]
            #
            # index=current_tag:[batch, 1]
            # [[tag1],
            #  [tag2],
            #  [tag3],
            # ]
            #
            # index of index:
            # [(0,0)
            #  (1,0)
            #  (2,0)
            # ...]
            #
            # dim=1,因此将index的值依次填index_of_index对应的dim=1上
            # new_index=
            # [(0,tag1)
            #  (1,tag2)
            #  (2,tag3)
            # ...]
            # 从logits[t]取new_index的值
            # [[s1],[s2],...]
            # emit_score_origin: [batch,1], 即每条样本它的state所对应的发射分数
            emit_score_origin = logits[t].gather(dim=1, index=current_tag.view(batch_size, 1))
            # emit_score: [batch]
            emit_score = emit_score_origin.squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            # score:[batch]
            # transition_score:[batch]
            # emit_score: [batch]
            # mask:[seq_len, batch] -> mask[t]:[batch]
            score = score + transition_score * mask[t + 1] + emit_score * mask[t]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        # mask:[seq_len, batch], 需要mask是因为同一个batch中每个样本的长度不一样
        # last_tag_index:[batch]
        last_tag_index = mask.sum(0).long() - 1
        # tags:[seq_len, batch]
        # tags:[batch], 获取每条样本最后的tag
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        # 上面的for循环并没有包含最终的last_tags的发射概率
        if self.include_start_end_transitions:
            # tags:[batch]
            # start_transitions:[num_tags], 初始分布
            # last_transition_score:[batch]
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        # logits:[seq_len, batch, num_tags], logits[-1]:[batch, num_tags]
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_emit_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_emit_score = last_emit_score.squeeze()  # (batch_size,)
        # score:[batch], 逐条样本将所有时间步的转移与发射概率相加
        score = score + last_transition_score + last_emit_score * mask[-1]

        return score

    def forward(
        self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """Computes the log likelihood for the given batch of input sequences $(x,y)$
        计算输入的序列logits+tags的概率

        Args:
            inputs (torch.Tensor): (batch_size, sequence_length, num_tags) tensor of logits for the inputs $x$
            tags (torch.Tensor): (batch_size, sequence_length) tensor of tags $y$
            mask (torch.BoolTensor, optional): (batch_size, sequence_length) tensor of masking flags.
                Defaults to None.

        Returns:
            torch.Tensor: (batch_size,) log likelihoods $log P(y|x)$ for each input
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            # The code below fails in weird ways if this isn't a bool tensor, so we make sure.
            mask = mask.to(torch.bool)

        # p(y|x;w) = exp(w*feat(xi,j,y(j-1),y(j)))/Z, xi指第i条样本,y为state, j为第j个时间步, Z为归一化因子
        # log p(y|x)
        # 求解分母,log_Z=log(sum(p(y|x;w))) over all y1~yt序列
        # inputs:[batch,seq_len, state_num]
        # transitions:[state_num, state_num]
        # mask:[batch, seq_len]
        # log_denominator:[batch]
        log_denominator = self._input_likelihood(inputs, self.transitions, mask) # 求解分母

        # 求解分子: w*feat(xi,j,y(j-1),y(j))
        # log_numerator:[batch]
        log_numerator = self._joint_likelihood(inputs, self.transitions, tags, mask)

        # result:[1], sum of log(p(y|x;w)) in batch, log(p|x;w)= w*feat(xj,j,y(j-1),y(j)) - Z
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(
        self, logits: torch.Tensor, mask: torch.BoolTensor = None, top_k: int = None
    ) -> Union[List[VITERBI_DECODING], List[List[VITERBI_DECODING]]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.

        Returns a list of results, of the same size as the batch (one result per batch member)
        Each result is a List of length top_k, containing the top K viterbi decodings
        Each decoding is a tuple  (tag_sequence, viterbi_score)

        For backwards compatibility, if top_k is None, then instead returns a flat list of
        tag sequences (the top tag sequence for each batch item).
        """
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)

        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False

        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.full((num_tags + 2, num_tags + 2), -10000.0, device=logits.device)

        # Apply transition constraints
        constrained_transitions = self.transitions * self._constraint_mask[
            :num_tags, :num_tags
        ] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[
                start_tag, :num_tags
            ] = self.start_transitions.detach() * self._constraint_mask[
                start_tag, :num_tags
            ].data + -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = self.end_transitions.detach() * self._constraint_mask[
                :num_tags, end_tag
            ].data + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (
                1 - self._constraint_mask[start_tag, :num_tags].detach()
            )
            transitions[:num_tags, end_tag] = -10000.0 * (
                1 - self._constraint_mask[:num_tags, end_tag].detach()
            )

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.empty(max_seq_length + 2, num_tags + 2, device=logits.device)

        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.0)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.0
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1 : (sequence_length + 1), :num_tags] = masked_prediction
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.0

            # We pass the tags and the transitions to `viterbi_decode`.
            viterbi_paths, viterbi_scores = util.viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transitions,
                top_k=top_k,
            )
            top_k_paths = []
            for viterbi_path, viterbi_score in zip(viterbi_paths, viterbi_scores):
                # Get rid of START and END sentinels and append.
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)

        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]

        return best_paths
