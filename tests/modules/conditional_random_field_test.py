import itertools
import math

from pytest import approx, raises
import torch
from numpy.testing import assert_allclose

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import (
    ConditionalRandomFieldWeightEmission,
    ConditionalRandomFieldWeightTrans,
    ConditionalRandomFieldWeightLannoy,
)
from allennlp.modules.conditional_random_field.conditional_random_field import allowed_transitions
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase


class TestConditionalRandomField(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        # logits:[batch, seq_len, state_num]
        self.logits = torch.Tensor(
            [
                [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
                [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
            ]
        )
        # tags:[batch, seq_len]
        self.tags = torch.LongTensor([[2, 3, 4], [3, 2, 2]])

        # state_num=N=5个状态的转换概率
        # transitions:[state_num, state_num]
        self.transitions = torch.Tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        # 初始时每个状态的概率，N=5
        # transitions_from_start:[state_num]
        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.6])
        # 结束时每个状态的概率，N=5，不是很懂为何有这个概率(hkx)
        # transitions_to_end:[state_num]
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4])

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = ConditionalRandomField(num_tags=5) # 即有5个状态转换
        # [state_num, state_num]
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        # [state_num,]
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        # [state_num,]
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def cal_score_of_numerator(self, logits, tags):
        """
        对于给定的序列，计算概率 log(p(y|x)), 但这里只计算了分子，没计算归一化的分母
        logits:[seq_len, state_num]
        tags:  [seq_len,]
        return: double

        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        # TODO:但我并不清楚为何最后的分布也需要指定？
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # 计算转移概率
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag] # 前后tag之间转移的分数
            
        # Add in the logits for the observed tags
        # logits:[seq_len, state_num],对应于CRF中各个时间步的 w*feat(x,j,s(j-1),s(j)), feat(*)为crf中全局特征
        # tags:[seq_len]
        # p(s|x;w) = exp(w*feat(x,j,s(j-1),s(j)))/Z, xi指第i条样本, s为state, j为第j个时间步, Z为归一化因子
        # 计算发射概率emit
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def naive_most_likely_sequence(self, logits, mask):
        # 手工验证viterbi解码的正确性
        # logits:[batch, seq_len, state_num]
        # mask:  [batch, seq_len]

        # We iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
       
        # 对于每一条样本，暴力找出最有可能的状态路径 
        # most_likely_tags:[batch, seq_len]
        # best_scores:[batch]
        most_likely_tags = []
        best_scores = []

        # logits:[batch, seq_len, state_num]
        # mask:  [batch, seq_len]
        for logit, mas in zip(logits, mask):
            # 找出每个seq序列中不被mask掉的logit
            mask_indices = mas.nonzero(as_tuple=False).squeeze() # 找出非0元素
            logit = torch.index_select(logit, 0, mask_indices) # 只取非0位置的score
            # logit:[seq_len, state_num]
            sequence_length = logit.shape[0]
            # state_num=5
            state_num=logit.shape[1]
            most_likely, most_likelihood = None, -float("inf")
            # product枚举所有state且长度为sequence_length的序列组合
            # 可枚举的序列长度是：state_num^seq_len
            """
            eg:
            >>> for x in itertools.product([0,1], repeat=3):
                    print(x)
                输出2^3=8种组合:
                (0, 0, 0)
                (0, 0, 1)
                (0, 1, 0)
                (0, 1, 1)
                (1, 0, 0)
                (1, 0, 1)
                (1, 1, 0)
                (1, 1, 1)
            """
            for tag_seq in itertools.product(range(state_num), repeat=sequence_length):
                #logit: [seq_len, state_num]
                #tag_seq: [seq_len, ]
                score = self.cal_score_of_numerator(logit.data, tag_seq)
                if score > most_likelihood:
                    most_likely, most_likelihood = tag_seq, score

            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        return most_likely_tags, best_scores

    def test_forward_works_without_mask(self):
        # inputs = logits:[batch, seq_len, state_num]
        # tags:[batch, seq_len]
        # log_likelihood:[batch,]
        # 计算batch中所有样本的log p(y|x)
        log_likelihood = self.crf.forward(inputs=self.logits, tags=self.tags).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood_in_batch = 0.0

        # For each instance, manually compute the numerator(分子)
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        
        # logits:[batch, seq_len, state_num]
        # tags:[batch, seq_len]
        # batch中的第i个样本
        # logits_i:[seq_len, state_num]
        # tags_i:[ seq_len]
        for logits_i, tags_i in zip(self.logits, self.tags):
            numerator = self.cal_score_of_numerator(logits_i.detach(), tags_i.detach())
            all_scores = [
                # logits_i:[seq_len, state_num]
                self.cal_score_of_numerator(logits_i.detach(), tags_j)
                # tags_j:所有5种状态*时间步的枚举,即所有可能路径序列的枚举, 5^3
                #(0, 0, 0)
                #(0, 0, 1)
                # ...
                for tags_j in itertools.product(range(5), repeat=3)
            ]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood_in_batch += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood_in_batch.item() == approx(log_likelihood)

    def test_forward_works_with_mask(self):
        # Use a non-trivial mask
        # mask:[batch, seq_len]
        mask = torch.tensor([[True, True, True],
                             [True, True, False]])
        # scalar
        log_likelihood = self.crf.forward(self.logits, self.tags, mask).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        # 第i条样本
        for logits_i, tags_i, mask_i in zip(self.logits, self.tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]

            numerator = self.cal_score_of_numerator(logits_i, tags_i)
            all_scores = [
                self.cal_score_of_numerator(logits_i, tags_j)
                for tags_j in itertools.product(range(5), repeat=sequence_length)
            ]
            # log_sum_exp(w*feat(x,yi-1,yi))
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_viterbi_tags(self):
        mask = torch.tensor([[True, True, True],
                             [True, False, True]])

        viterbi_path = self.crf.viterbi_tags(self.logits, mask)

        # Separate the tags and scores.
        viterbi_tags = [x for x, y in viterbi_path]
        viterbi_scores = [y for x, y in viterbi_path]

        most_likely_tags, best_scores = self.naive_most_likely_sequence(self.logits, mask)

        assert viterbi_tags == most_likely_tags
        assert_allclose(viterbi_scores, best_scores, rtol=1e-5)

    def test_viterbi_tags_no_mask(self):
        # logits:[batch, seq_len, num_tags]
        # # list of tags, and a viterbi score, eg:[(tag_seq, score)]
        viterbi_path = self.crf.viterbi_tags(self.logits)

        # Separate the tags and scores.
        viterbi_tags = [x for x, y in viterbi_path]
        viterbi_scores = [y for x, y in viterbi_path]

        mask = torch.tensor([[True, True, True], [True, True, True]])
        most_likely_tags, best_scores = self.naive_most_likely_sequence(self.logits, mask)

        assert viterbi_tags == most_likely_tags
        assert_allclose(viterbi_scores, best_scores, rtol=1e-5)

    def test_viterbi_tags_top_k(self):
        mask = torch.tensor([[True, True, True], [True, True, False]])

        best_paths = self.crf.viterbi_tags(self.logits, mask, top_k=2)

        # Ensure the top path matches not passing top_k
        top_path_and_score = [top_k_paths[0] for top_k_paths in best_paths]
        assert top_path_and_score == self.crf.viterbi_tags(self.logits, mask)

        next_path_and_score = [top_k_paths[1] for top_k_paths in best_paths]
        next_viterbi_tags = [x for x, _ in next_path_and_score]

        # Check that the next best viterbi tags are what I think they should be.
        assert next_viterbi_tags == [[4, 2, 3], [3, 2]]

    def test_constrained_viterbi_tags(self):
        constraints = {
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 0),
        }

        # Add the transitions to the end tag
        # and from the start tag.
        for i in range(5):
            constraints.add((5, i))
            constraints.add((i, 6))

        crf = ConditionalRandomField(num_tags=5, constraints=constraints)
        crf.transitions = torch.nn.Parameter(self.transitions)
        crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        mask = torch.tensor([[True, True, True], [True, True, False]])

        viterbi_path = crf.viterbi_tags(self.logits, mask)

        # Get just the tags from each tuple of (tags, score).
        viterbi_tags = [x for x, y in viterbi_path]

        # Now the tags should respect the constraints
        assert viterbi_tags == [[2, 3, 3], [2, 3]]

    def test_allowed_transitions(self):

        bio_labels = ["O", "B-X", "I-X", "B-Y", "I-Y"]  # start tag, end tag
        #              0     1      2      3      4         5          6
        allowed = allowed_transitions("BIO", dict(enumerate(bio_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == {  # Extra column for end tag.
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 6),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 6),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 6),
            (3, 0),
            (3, 1),
            (3, 3),
            (3, 4),
            (3, 6),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 6),
            (5, 0),
            (5, 1),
            (5, 3),  # Extra row for start tag
        }

        bioul_labels = [
            "O",
            "B-X",
            "I-X",
            "L-X",
            "U-X",
            "B-Y",
            "I-Y",
            "L-Y",
            "U-Y",
        ]  # start tag, end tag
        #                0     1      2      3      4      5      6      7      8          9        10
        allowed = allowed_transitions("BIOUL", dict(enumerate(bioul_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == {  # Extra column for end tag.
            (0, 0),
            (0, 1),
            (0, 4),
            (0, 5),
            (0, 8),
            (0, 10),
            (1, 2),
            (1, 3),  # noqa
            (2, 2),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 4),
            (3, 5),
            (3, 8),
            (3, 10),
            (4, 0),
            (4, 1),
            (4, 4),
            (4, 5),
            (4, 8),
            (4, 10),
            (5, 6),
            (5, 7),
            (6, 6),
            (6, 7),
            (7, 0),
            (7, 1),
            (7, 4),
            (7, 5),
            (7, 8),
            (7, 10),
            (8, 0),
            (8, 1),
            (8, 4),
            (8, 5),
            (8, 8),
            (8, 10),
            # Extra row for start tag.
            (9, 0),
            (9, 1),
            (9, 4),
            (9, 5),
            (9, 8),
        }

        iob1_labels = ["O", "B-X", "I-X", "B-Y", "I-Y"]  # start tag, end tag
        #              0     1      2      3      4         5          6
        allowed = allowed_transitions("IOB1", dict(enumerate(iob1_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == {  # Extra column for end tag.
            (0, 0),
            (0, 2),
            (0, 4),
            (0, 6),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 6),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 6),
            (3, 0),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 6),
            (4, 0),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 6),
            (5, 0),
            (5, 2),
            (5, 4),  # Extra row for start tag
        }
        with raises(ConfigurationError):
            allowed_transitions("allennlp", {})

        bmes_labels = ["B-X", "M-X", "E-X", "S-X", "B-Y", "M-Y", "E-Y", "S-Y"]  # start tag, end tag
        #               0      1      2      3      4      5      6      7       8          9
        allowed = allowed_transitions("BMES", dict(enumerate(bmes_labels)))
        assert set(allowed) == {
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),  # Extra column for end tag.
            (2, 0),
            (2, 3),
            (2, 4),
            (2, 7),
            (2, 9),  # noqa
            (3, 0),
            (3, 3),
            (3, 4),
            (3, 7),
            (3, 9),
            (4, 5),
            (4, 6),
            (5, 5),
            (5, 6),
            (6, 0),
            (6, 3),
            (6, 4),
            (6, 7),
            (6, 9),
            (7, 0),
            (7, 3),
            (7, 4),
            (7, 7),
            (7, 9),
            (8, 0),
            (8, 3),
            (8, 4),
            (8, 7),  # Extra row for start tag
        }


class TestConditionalRandomFieldWeightEmission(TestConditionalRandomField):
    def setup_method(self):
        super().setup_method()

        self.label_weights = torch.FloatTensor([1.0, 1.0, 0.5, 0.5, 0.5])

        self.crf = ConditionalRandomFieldWeightEmission(5, label_weights=self.label_weights)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        # Use the CRF Module with labels weights.
        self.crf.label_weights = torch.nn.Parameter(self.label_weights, requires_grad=False)

    def score_with_weights(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits, the transition weights in the CRF model
        and the label weights.
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag] * self.label_weights[tag]
        return total

    def test_forward_works_without_mask(self):
        log_likelihood = self.crf(self.logits, self.tags).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in zip(self.logits, self.tags):
            numerator = self.score_with_weights(logits_i.detach(), tags_i.detach())
            all_scores = [
                self.score_with_weights(logits_i.detach(), tags_j)
                for tags_j in itertools.product(range(5), repeat=3)
            ]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_forward_works_with_mask(self):
        # Use a non-trivial mask
        mask = torch.tensor([[True, True, True], [True, True, False]])

        log_likelihood = self.crf(self.logits, self.tags, mask).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i, mask_i in zip(self.logits, self.tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]

            numerator = self.score_with_weights(logits_i, tags_i)
            all_scores = [
                self.score_with_weights(logits_i, tags_j)
                for tags_j in itertools.product(range(5), repeat=sequence_length)
            ]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)


class TestConditionalRandomFieldWeightTrans(TestConditionalRandomFieldWeightEmission):
    def setup_method(self):
        super().setup_method()

        self.label_weights = torch.FloatTensor([1.0, 1.0, 0.5, 0.5, 0.5])

        self.crf = ConditionalRandomFieldWeightTrans(5, label_weights=self.label_weights)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        # Use the CRF Module with labels weights.
        self.crf.label_weights = torch.nn.Parameter(self.label_weights, requires_grad=False)

    def score_with_weights(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits, the transition weights in the CRF model
        and the label weights.
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag] * self.label_weights[tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag] * self.label_weights[tag]
        return total


class TestConditionalRandomFieldWeightLannoy(TestConditionalRandomFieldWeightEmission):
    def setup_method(self):
        super().setup_method()

        self.label_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0])

        self.crf = ConditionalRandomFieldWeightLannoy(5, label_weights=self.label_weights)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        # Use the CRF Module with labels weights.
        self.crf.label_weights = torch.nn.Parameter(self.label_weights, requires_grad=False)
