# nohup python main.py --phase train --model_file models/wikiqa_config/drmm_wikiqa.config > ./logs/log.drmm.1214 &
# python main.py --phase predict --model_file models/wikiqa_config/drmm_wikiqa.config

nohup python main.py --phase train --model_file models/wikiqa_config/bimpm_wikiqa.config > ./logs/log.bimpm.1214 &
# python main.py --phase predict --model_file models/wikiqa_config/bimpm_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/eattention_wikiqa.config > ./logs/log.eattention.1214 &
# python main.py --phase predict --model_file models/wikiqa_config/eattention_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/elstmattention_wikiqa.config > ./logs/log.elstmattention.1214 &
# python main.py --phase predict --model_file models/wikiqa_config/elstmattention_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/mergedattention_wikiqa.config > ./logs/log.mergedattention.1211 &
# python main.py --phase predict --model_file models/wikiqa_config/merged_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/bigru_wikiqa.config > ./logs/log.bigru.1211 &
# python main.py --phase predict --model_file models/wikiqa_config/bigru_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/multibigru_wikiqa.config > ./logs/log.multibigru.1211 &
# python main.py --phase predict --model_file models/wikiqa_config/multibigru_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/bilstm_wikiqa.config > ./logs/log.bilstm.1211 &
# python main.py --phase predict --model_file models/wikiqa_config/bilstm_wikiqa.config

# nohup python main.py --phase train --model_file models/wikiqa_config/multibilstm_wikiqa.config > ./logs/log.multibilstm.1211 &
# python main.py --phase predict --model_file models/wikiqa_config/multibilstm_wikiqa.config
