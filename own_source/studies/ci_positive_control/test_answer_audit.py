import unittest
from audit_expansion import approval


class AnswerAuditTest(unittest.TestCase):
    def test_binary_substring_is_not_an_answer(self):
        self.assertFalse(approval('No','Not Applicable',['result']).startswith('APPROVED'))

    def test_identifier_digits_are_preserved(self):
        self.assertFalse(approval('PAY_2','PAY_0',['PAY_0: .51; PAY_2: .10']).startswith('APPROVED'))

    def test_composite_identifier_is_not_bag_of_words(self):
        self.assertFalse(approval('other_rated_categories','other rated categories',['other rated categories']).startswith('APPROVED'))

    def test_correct_numeric_format_supported_by_real_output(self):
        self.assertTrue(approval('2020.0','2020',['Latest year: 2020.0']).startswith('APPROVED'))
        self.assertTrue(approval('Sales department with 4140 employees',
            'The sales department has 4,140 employees.',['Sales department: 4140 employees']).startswith('APPROVED'))

    def test_numeric_rounding_is_not_silently_accepted(self):
        self.assertFalse(approval('8.89663104713','8.896',['8.89663104713']).startswith('APPROVED'))

    def test_unresolved_execution_error_is_quarantined(self):
        self.assertFalse(approval('0.0','0.0',['ModuleNotFoundError: no module named statsmodels']).startswith('APPROVED'))

    def test_ambiguous_category_comparison_is_not_approved(self):
        self.assertFalse(approval('Hierarchical','K-means is better compared to Hierarchical',
            ['K-means .3683; Hierarchical .3918']).startswith('APPROVED'))

    def test_units_and_order_remain_material(self):
        self.assertFalse(approval('5 mg','5 g',['5 mg']).startswith('APPROVED'))
        self.assertFalse(approval('A, B','B, A',['A, B']).startswith('APPROVED'))


if __name__=='__main__':unittest.main()
