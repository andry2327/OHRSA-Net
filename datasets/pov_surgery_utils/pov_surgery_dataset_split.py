class PovSurgerySplits:
    def __init__(self):
        self.DATASET_ENTRIES_NAMES = [
            "R2_d_diskplacer_1", "d_diskplacer_2", "m_friem_2", "r_scalpel_2",
            "R2_d_diskplacer_2", "d_friem_1", "m_scalpel_1", "r_scalpel_3",
            "R2_d_friem_1", "d_friem_2", "m_scalpel_2", "r_scalpel_4",
            "R2_d_friem_2", "d_scalpel_1", "r_diskplacer_1", "s_diskplacer_1",
            "R2_d_scalpel_1", "d_scalpel_2", "r_diskplacer_2", "s_diskplacer_2",
            "R2_i_diskplacer_1", "i_diskplacer_1", "r_diskplacer_3", "s_friem_1",
            "R2_r_diskplacer_1", "i_diskplacer_2", "r_diskplacer_4", "s_friem_2",
            "R2_r_friem_1", "i_friem_1", "r_diskplacer_5", "s_friem_3",
            "R2_r_scalpel_1", "i_friem_2", "r_diskplacer_6", "s_scalpel_1",
            "R2_r_scalpel_2", "i_scalpel_1", "r_friem_1", "s_scalpel_2",
            "R2_s_diskplacer_1", "i_scalpel_2", "r_friem_2", "s_scalpel_3",
            "R2_s_friem_1", "m_diskplacer_1", "r_friem_3", "s_scalpel_4",
            "R2_s_scalpel_1", "m_diskplacer_2", "r_friem_4",
            "d_diskplacer_1", "m_friem_1", "r_scalpel_1"
            ]

        # Splits from POV-Surgery paper and code
        self.TEST_LIST = ['m_diskplacer_1', 'i_scalpel_1', 'd_friem_1', 'r_friem_3', 'R2_d_diskplacer_1',
                          'R2_r_scalpel_1', 'R2_d_friem_1', 'R2_d_scalpel_1', 'R2_r_scalpel_2', 'R2_s_scalpel_1',
                          'R2_d_friem_2', 'R2_r_friem_1', 'R2_s_friem_1', 'R2_d_diskplacer_2', 'R2_r_diskplacer_1',
                          'R2_s_diskplacer_1', 'R2_i_diskplacer_1']

        self.TRAIN_LIST = ['r_diskplacer_1', 'm_diskplacer_2', 'r_diskplacer_2', 'r_diskplacer_3',
                           'r_diskplacer_4', 's_diskplacer_1', 'm_friem_1', 'm_friem_2', 'r_friem_1', 'r_friem_2', 's_friem_1', 's_friem_2',
                           'm_scalpel_1', 'm_scalpel_2', 'r_scalpel_1', 'r_scalpel_2', 's_scalpel_1', 's_scalpel_2',
                           'd_diskplacer_1', 'i_diskplacer_1', 'r_diskplacer_5', 'd_scalpel_1', 'r_scalpel_3',
                           's_scalpel_3', 'd_diskplacer_2', 'i_diskplacer_2', 'r_diskplacer_6', 's_diskplacer_2', 'd_friem_2',
                           'i_friem_2', 'r_friem_4', 's_friem_3', 'd_scalpel_2', 'i_scalpel_2', 'r_scalpel_4']
    
    def get_splits(self):
        return self.TRAIN_LIST, self.TEST_LIST