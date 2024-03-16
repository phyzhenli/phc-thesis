import pandas as pd
import matplotlib.pyplot as plt

def PDA(df):
    df.loc['PDA'] = df.loc['Total cell area(um^2)'] * df.loc['Total power(mW)'] * df.loc['data arrival time(ps)'] / 1000 * 1000 # ps -> ns; mW -> uW


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__ == '__main__':

    XWYF_2GHz = pd.read_csv('/home/zli/Documents/PhD/Approximate_Computing/AC_ASIC/VEAM/Unsigned/DNNs/MNIST/LeNet/XWYF/2GHz.csv', index_col=0)

    XWYF_Accuracy = pd.read_csv('/home/zli/Documents/PhD/Approximate_Computing/AC_ASIC/VEAM/Unsigned/DNNs/MNIST/LeNet/XWYF/Accuracy.csv', index_col=0)
    XWYF_Accuracy['ubit8_designware'] = 99.41

    SC_XWYF_ADP = pd.read_csv('/home/zli/Documents/PhD/Approximate_Computing/AC_ASIC/VEAM/Unsigned/DNNs/MNIST/LeNet/XWYF/SC_LeNet_XWYF.csv', index_col=0)
    SC_XWYF_ADP['ADP'] = SC_XWYF_ADP['ADP'] / 1000 # ps -> ns
    SC_XWYF_ADP = SC_XWYF_ADP.T
    for index, row in SC_XWYF_ADP.iteritems(): # 列遍历
        if 'ubit8_designware' in index:
            SC_ubit8_designware_ADP_Accuracy = pd.concat([ # 行拼接
                pd.DataFrame(SC_XWYF_ADP, columns=[index]),
                pd.DataFrame(columns=['./outputs/SC_ubit8_designware/systolic_cube_without_fifo'], index=['Accuracy'], data=[99.41])
            ])
    # print(SC_ubit8_designware_ADP_Accuracy)
    SC_XWYF_ADP.drop('./outputs/SC_ubit8_designware/systolic_cube_without_fifo', axis=1, inplace=True)
    
    ubit8_designware_2GHz_Accuracy = pd.concat([
        pd.DataFrame(XWYF_2GHz, columns=['ubit8_designware']),
        pd.DataFrame(XWYF_Accuracy, columns=['ubit8_designware'])
    ])
    PDA(ubit8_designware_2GHz_Accuracy)

    XWYF_2GHz.drop(['ubit8_designware'], axis=1, inplace=True)
    XWYF_Accuracy.drop(['ubit8_designware'], axis=1, inplace=True)
    XWYF_2GHz_Accuracy = pd.concat([XWYF_2GHz, XWYF_Accuracy])
    PDA(XWYF_2GHz_Accuracy)

    SC_XWYF_ADP.loc['Accuracy'] = [None] * len(SC_XWYF_ADP.columns) # 添加一行
    for SC_col, SC_row in SC_XWYF_ADP.iteritems():
        for mul_col, mul_row in XWYF_2GHz_Accuracy.iteritems():
            if mul_col == SC_col.split('/')[2][3:]:
                SC_XWYF_ADP[SC_col]['Accuracy'] = XWYF_2GHz_Accuracy[mul_col]['Accuracy']

    fig = plt.figure()
    ax = fig.add_subplot()

    XWYF_2GHz_Accuracy.sort_values(by='PDA', axis='columns', inplace=True)
    # print(XWYF_2GHz_Accuracy)
    XWYF_Pareto_mul_name = []
    for index, row in XWYF_2GHz_Accuracy.iteritems():
        flag = True
        for p_XWYF in XWYF_Pareto_mul_name:
            if XWYF_2GHz_Accuracy[p_XWYF]['Accuracy'] >= XWYF_2GHz_Accuracy[index]['Accuracy']:
                flag = False
        if flag:
            XWYF_Pareto_mul_name.append(index)

    XWYF_2GHz_Accuracy_Pareto = pd.DataFrame()
    SC_XWYF_ADP_mulPareto = pd.DataFrame()
    for P_mul_name in XWYF_Pareto_mul_name:
        XWYF_2GHz_Accuracy_Pareto[P_mul_name] = XWYF_2GHz_Accuracy[P_mul_name]
        XWYF_2GHz_Accuracy.drop([P_mul_name], axis=1, inplace=True)
        for col, index in SC_XWYF_ADP.iteritems():
            if P_mul_name == col.split('/')[2][3:]:
                SC_XWYF_ADP_mulPareto[col] = SC_XWYF_ADP[col]
                SC_XWYF_ADP.drop(col, axis=1, inplace=True)
                break
    # print(XWYF_2GHz_Accuracy_Pareto)
    # print(SC_XWYF_ADP_mulPareto)


    DesignW_corlor='red'
    not_Pareto_color = 'C4'
    Pareto_color = 'C1'
    ss=50

    ax.scatter(SC_ubit8_designware_ADP_Accuracy['./outputs/SC_ubit8_designware/systolic_cube_without_fifo']['Accuracy'], SC_ubit8_designware_ADP_Accuracy['./outputs/SC_ubit8_designware/systolic_cube_without_fifo']['ADP'], c=DesignW_corlor, s=ss*3, marker='*', label="SC with DesignW", zorder=90)

    ax.scatter(SC_XWYF_ADP_mulPareto.loc['Accuracy'], SC_XWYF_ADP_mulPareto.loc['ADP'], c=Pareto_color, s=ss*3, marker='^', label="SC with XWYF (Pareto)", zorder=91)

    ax.scatter(SC_XWYF_ADP.loc['Accuracy'], SC_XWYF_ADP.loc['ADP'], c=not_Pareto_color, s=ss*3, marker='^', label="SC with XWYF (not Pareto)", zorder=90)
    
    ax.grid(linestyle='--')
    ax.legend()

    ax.set_xlabel(r'$Accuracy\ (\%)$')
    ax.set_ylabel(r'$ADP\ (\mu m^2 \cdot ns)$')

    plt.savefig('SC_LeNet_XWYF_ADP.pdf', bbox_inches = 'tight')
    plt.show()