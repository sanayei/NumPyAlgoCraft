from collections import defaultdict
from itertools import combinations

# ----------------------------
#   Apriori Algorithm
# ----------------------------


class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.association_rules = []

    def fit(self, transactions):
        self.transactions = list(map(set, transactions))
        self.num_transactions = len(self.transactions)
        self.frequent_itemsets = []
        k = 1
        current_lk = self.find_frequent_itemsets(k)

        while current_lk:
            self.frequent_itemsets.append(current_lk)
            k += 1
            candidate_ck = self.apriori_gen(current_lk)
            current_lk = self.find_frequent_itemsets(k, candidate_ck)

        self.generate_association_rules()

    def find_frequent_itemsets(self, k, candidate_ck=None):
        if k == 1:
            item_counts = defaultdict(int)
            for transaction in self.transactions:
                for item in transaction:
                    item_counts[frozenset([item])] += 1
            current_lk = {
                itemset
                for itemset, count in item_counts.items()
                if count / self.num_transactions >= self.min_support
            }
        else:
            item_counts = defaultdict(int)
            for transaction in self.transactions:
                for candidate in candidate_ck:
                    if candidate.issubset(transaction):
                        item_counts[candidate] += 1
            current_lk = {
                itemset
                for itemset, count in item_counts.items()
                if count / self.num_transactions >= self.min_support
            }
        return current_lk

    def apriori_gen(self, current_lk):
        candidates = set()
        current_lk_list = list(current_lk)
        for i in range(len(current_lk_list)):
            for j in range(i + 1, len(current_lk_list)):
                l1 = list(current_lk_list[i])
                l2 = list(current_lk_list[j])
                l1.sort()
                l2.sort()
                if l1[:-1] == l2[:-1]:
                    candidate = current_lk_list[i].union(current_lk_list[j])
                    if len(candidate) == len(current_lk_list[i]) + 1:
                        candidates.add(candidate)
        return candidates

    def generate_association_rules(self):
        for k, itemset_level in enumerate(self.frequent_itemsets, start=1):
            if k < 2:
                continue
            for itemset in itemset_level:
                subsets = [
                    frozenset(x)
                    for i in range(1, len(itemset))
                    for x in combinations(itemset, i)
                ]
                for antecedent in subsets:
                    consequent = itemset - antecedent
                    if consequent:
                        support_itemset = self.get_support(itemset)
                        support_antecedent = self.get_support(antecedent)
                        confidence = support_itemset / support_antecedent
                        if confidence >= self.min_confidence:
                            rule = (antecedent, consequent, support_itemset, confidence)
                            self.association_rules.append(rule)

    def get_support(self, itemset):
        count = sum(
            1 for transaction in self.transactions if itemset.issubset(transaction)
        )
        return count / self.num_transactions

    def get_frequent_itemsets(self):
        return self.frequent_itemsets

    def get_association_rules(self):
        return self.association_rules
