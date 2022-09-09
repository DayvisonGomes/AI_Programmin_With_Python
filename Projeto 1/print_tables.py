# PROGRAMMER: Dayvison Gomes de Oliveira
# DATE CREATED: 11/08/2022                                  
# REVISED DATE: 12/08/2022

import pandas as pd

def print_tables(results_stats):
    
     
    filenames = ['alexnet_pet-images.txt','resnet_pet-images.txt','vgg_pet-images.txt']
    
    try:
        for file in filenames:
            arquivo = open(file)
            arquivo.close()
             
        table_pct = pd.DataFrame(data={'CNN Model': [], '% Dogs Correct':[], '% Not-a-Dog Correct':[], '% Breeds Correct':[], '% Match Labels':[]})
        names_pct = ['pct_correct_dogs', 'pct_correct_notdogs', 'pct_correct_breed','pct_label_matches']

        for file in filenames:
            pct = []

            with open(file) as f:
                arq = f.readlines()

                for line in arq:
                    line_ = line.strip()

                    for name_pct in names_pct:
                        if name_pct in line_:
                            pct.append(line_.split(':')[1][1:])

            table_pct.loc[len(table_pct)] = [ file.split('_')[0], pct[0], pct[1], pct[2], pct[3] ]
        
        df_numbers = pd.DataFrame(data={'#':['Total Images','Dog Images','Not-a-Dog Images'], 'Count':[results_stats['n_images'],results_stats['n_dogs_img'],results_stats['n_notdogs_img']]})
        print("\nFinal tables: \n")
        print(df_numbers.set_index('#'))
        print('')
        print(table_pct.set_index('CNN Model'))
        print('')
    
    
    except:
        print('Run the file run_models_batch.sh first')
        
        return None
    
    return None
        
    