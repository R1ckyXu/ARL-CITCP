import pandas as pd
import decimal

decimal.getcontext().rounding = "ROUND_HALF_UP"

df = pd.read_csv("./results_csv/add_result.csv")
saveFileName = 'ARL-CITCP.xlsx'


def analysis_noth(df):
    lista = []
    relist = ["ATCF5"]
    exlist = ["Dataset", "NAPFD", "TTF", "Durations"]

    for column, env in enumerate(sorted(df['env'].unique())):
        env1 = env.title()
        env1 = env1.replace("_", " ")
        if env1 == "Iofrol":
            env1 = "IOF/ROL"
        slist = [env1]

        for title in ['napfd', 'ttf', 'durations']:
            for re in relist:
                tdf = df[df["env"].isin([env]) &
                         df["rewardfun"].isin([re])
                         ]

                slist.append(float(decimal.Decimal(str(tdf[title]).split()[1]).quantize(
                    decimal.Decimal("0.0000"))))

        lista.append(slist)

    slist = ['Average']
    rownums = len(lista)
    colnums = len(lista[0])

    for i in range(1, colnums):
        cc = 0
        for j in range(0, rownums):
            cc += lista[j][i]
        slist.append(float(decimal.Decimal(cc / rownums).quantize(
            decimal.Decimal("0.0000"))))
    lista.append(slist)

    print(exlist)
    cdf = pd.DataFrame(lista, columns=exlist)

    cdf.to_excel(saveFileName, index=False)


analysis_noth(df)
