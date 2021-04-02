import numpy as np
import editdistance

class Utils:
    @staticmethod
    def normalize(d, total):
        """
        Converting raw counts to frequencies
        :param d: collection of counts to be normalized
        :param total: normalization constant
        :return: normalized version of the input
        """

        obj_type = type(d)
        dn = obj_type()
        for key in d.keys():
            dn[key] = float(d[key]) / total
        return dn

    @staticmethod
    def editops(src_list, trg_list,look_error=1):

        chart = np.zeros((len(src_list) + 1, len(trg_list) + 1))
        bp = np.zeros((len(src_list) + 1, len(trg_list) + 1))
        # ins = 1, sub = 2, del = 3
        # down = delete, right = insert

        for j in range(1, len(trg_list) + 1):
            chart[0][j] = j
            bp[0][j] = 1

        for i in range(1, len(src_list) + 1):
            chart[i][0] = i
            bp[i][0] = 3

        for i in range(1, len(src_list) + 1):
            for j in range(1, len(trg_list) + 1):
                if src_list[i - 1] == trg_list[j - 1]:
                    chart[i][j] = chart[i - 1][j - 1]
                    bp[i][j] = 0
                else:
                    chart[i][j] = chart[i - 1][j - 1] + 1
                    bp[i][j] = 2
                    if chart[i][j - 1] + 1 < chart[i][j]:
                        chart[i][j] = chart[i][j - 1] + 1
                        bp[i][j] = 1
                    if chart[i - 1][j] + 1 < chart[i][j]:
                        chart[i][j] = chart[i - 1][j] + 1
                        bp[i][j] = 3

        i = len(src_list)
        j = len(trg_list)
        ops = []

        # assert chart[i][j] == editdistance.eval(src_list, trg_list)
        while i > 0 or j > 0:
            if bp[i, j] == 1:
                if look_error==1:
                    ops.append(("insert", "INS->{}".format(trg_list[j - 1])))
                j -= 1
            elif bp[i, j] == 2:
                if look_error==2:
                    ops.append(("replace", "{}->{}".format(src_list[i - 1], trg_list[j - 1])))
                j -= 1
                i -= 1
            elif bp[i, j] == 3:
                if look_error==3:
                    ops.append(("delete", "{}->DEL".format(src_list[i - 1])))
                i -= 1
            else:
                j -= 1
                i -= 1

        return ops[::-1]



