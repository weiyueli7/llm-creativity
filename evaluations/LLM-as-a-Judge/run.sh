
# single agent
python eval.py --prefix Results_v3/Single_Agent/Output

# llm discussion
python eval.py --prefix Results_v3/SciFi-Dis/Output/multi_agent/SciFi-Dis_multi_teacher_roleplay_3_5_meta-llama

# llm teacher
python eval.py --prefix Results_v4/SciFi-Teacher/Output/multi_agent/SciFi-Teacher_multi_teacher_roleplay_4_5_meta-llama


# llm review
python eval.py --prefix Results_v4/SciFi-Review/Output/multi_agent/SciFi-Review_multi_teacher_roleplay_3_5_meta-llama


# Get the results
python scoring.py
