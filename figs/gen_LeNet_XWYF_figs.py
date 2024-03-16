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

    ubit8_designware_2GHz_Accuracy = pd.concat([
        pd.DataFrame(XWYF_2GHz, columns=['ubit8_designware']),
        pd.DataFrame(XWYF_Accuracy, columns=['ubit8_designware'])
    ])
    PDA(ubit8_designware_2GHz_Accuracy)

    XWYF_2GHz.drop(['ubit8_designware'], axis=1, inplace=True)
    XWYF_Accuracy.drop(['ubit8_designware'], axis=1, inplace=True)
    XWYF_2GHz_Accuracy = pd.concat([XWYF_2GHz, XWYF_Accuracy])
    PDA(XWYF_2GHz_Accuracy)

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
    for P_mul_name in XWYF_Pareto_mul_name:
        XWYF_2GHz_Accuracy_Pareto[P_mul_name] = XWYF_2GHz_Accuracy[P_mul_name]
        XWYF_2GHz_Accuracy.drop([P_mul_name], axis=1, inplace=True)

    print(XWYF_2GHz_Accuracy_Pareto)

    DesignW_corlor='red'
    not_Pareto_color = 'C4'
    Pareto_color = 'C1'
    ss=50

    ax.scatter(ubit8_designware_2GHz_Accuracy['ubit8_designware']['Accuracy'], ubit8_designware_2GHz_Accuracy['ubit8_designware']['PDA'], c=DesignW_corlor, s=ss*3, marker='*', label="DesignW", zorder=90)

    ax.scatter(XWYF_2GHz_Accuracy_Pareto.loc['Accuracy'], XWYF_2GHz_Accuracy_Pareto.loc['PDA'], c=Pareto_color, s=ss*3, marker='^', label="XWYF (Pareto)", zorder=91)   
                
    ax.scatter(XWYF_2GHz_Accuracy.loc['Accuracy'], XWYF_2GHz_Accuracy.loc['PDA'], c=not_Pareto_color, s=ss*3, marker='^', label="XWYF (not Pareto)", zorder=90)
    
    ax.grid(linestyle='--')
    ax.legend()

    ax.set_xlabel(r'$Accuracy\ (\%)$')
    ax.set_ylabel(r'$PDA\ (\mu W \cdot ns \cdot \mu m^2)$')

    plt.savefig('LeNet_XWYF_PDA.pdf', bbox_inches = 'tight')
    plt.show()