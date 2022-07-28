from deberta_base_functions import *


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df['label'].values
        preds = oof_df[['pred_0', 'pred_1', 'pred_2']].values.tolist()
        score = get_score(preds, labels)
        LOGGER.info(f'Score: {score:<.4f}')


    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR + 'oof_df.pkl')
        oof_df.to_csv(OUTPUT_DIR + f'oof_df.csv', index=False)

    # if CFG.wandb:
    #     wandb.finish()

    A = pd.read_csv(OUTPUT_DIR + 'oof_df.csv')
    A.head()
