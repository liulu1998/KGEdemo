import os
from main import read_triples, read_dict, DATA_DIR


def get_train_val_entities(entity2id: dict):
    entities = set()
    with open(os.path.join(DATA_DIR, "train.txt"), 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            h_id, t_id = entity2id[h], entity2id[t]
            entities.add(h_id)
            entities.add(t_id)

    missing_cnt = 0

    with open(os.path.join(DATA_DIR, "valid_clean.txt"), 'w') as wf:
        with open(os.path.join(DATA_DIR, "valid.txt"), 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                h_id, t_id = entity2id[h], entity2id[t]
                if (h_id not in entities) or (t_id not in entities):
                    missing_cnt += 1
                    continue
                else:
                    wf.write(line)

    with open(os.path.join(DATA_DIR, "test_clean.txt"), 'w') as wf:
        with open(os.path.join(DATA_DIR, "test.txt"), 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                h_id, t_id = entity2id[h], entity2id[t]
                if (h_id not in entities) or (t_id not in entities):
                    missing_cnt += 1
                    continue
                else:
                    wf.write(line)
    print(missing_cnt)


if __name__ == '__main__':
    entity2id = read_dict("entities.dict")
    relation2id = read_dict("relations.dict")

    get_train_val_entities(entity2id)
