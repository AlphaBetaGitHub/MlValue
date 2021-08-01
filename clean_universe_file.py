import pandas as pd

from sparta.tomer.alpha_go.consts import LOCAL_PATH, US_UNIVERSE_FILE_NAME


def main():
    columns = ['Period', 'Name', 'Ticker', 'ISIN', 'Country', 'RBICS_ECON']
    # get universe
    with open(LOCAL_PATH + 'us_universe.txt') as f:
        end_file = []
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            if len(line) == 6:
                pass
            elif len(line) == 7:
                line.pop(2)
            elif len(line) == 8:
                line.pop(2)
                line.pop(2)

            date, name, ticker, isin, cc, industry = line[0], str(line[1]), str(line[2]), line[3], str(line[4]), str(line[5])
            end_file.append([date, name, ticker, isin, cc, industry])

            if len(end_file) % 100000 == 0:
                print('Procceed 100K rows')

    df = pd.DataFrame(end_file, columns=columns)
    df = df.iloc[1:]

    writer = pd.ExcelWriter(LOCAL_PATH + US_UNIVERSE_FILE_NAME, engine='xlsxwriter')

    df.to_excel(writer)
    writer.save()


if __name__ == '__main__':
    import fire

    fire.Fire(main)
