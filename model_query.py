import ktrain 
from ktrain import text as txt

class Models:
    def train_model(self, name, transformer_model, trn, val, preproc):

        print(name.center(50, '-'))

        model= txt.sequence_tagger('bilstm-bert',
                                preproc, verbose=0,
                                transformer_model=transformer_model)

        learner = ktrain.get_learner(model,
                                    train_data=trn,
                                    val_data=val,
                                    batch_size=128)

        learner.fit(0.01, 1, cycle_len=5,
                    checkpoint_folder=f'/tmp/saved_weights_{name}')

        predictor = ktrain.get_predictor(learner.model, preproc)

        return predictor

    def gen_predictors(self, models, train_filepath, val_filepath):

        (trn, val, preproc) = txt.entities_from_conll2003(train_filepath,
                                                        val_filepath=val_filepath,
                                                        verbose=0)
        predictors = []
        for (model, source) in models :
            predictors.append(self.train_model(name=model,
                                        transformer_model=source,
                                        trn=trn, val=val,
                                        preproc=preproc))
        return predictors

    def get_agreements(self, texts,predictions,unlabeled_ids) :

        agreements =[]
        entities = [self.get_entities_from_prediction(prediction) for prediction in predictions]

        for i in range(len(texts)):
            veredicts = {}
            majority    = 0;
            majority_id = 0;

            for j in range(len(predictions)):
                veredicts[j]=0
                for k in range(len(predictions)):
                    if(entities[j][i] == entities[k][i]):
                        veredicts[j]+=1

            for v in veredicts:        
                if veredicts[v] > majority :
                    majority = veredicts[v]
                    majority_id = v

            if majority > len(predictions)//2:
                agreements.append({'id':unlabeled_ids[i],
                                'text':texts[i],
                                'prediction': predictions[majority_id][i],
                                'model_version':'concordancia'})

        return agreements

    def get_entities_from_prediction(self, predictions):

        predicted_entities = []

        for pred in predictions:
            entities = {}
            entity = []
            prev_label = 'O'
            for token, iob in pred:
                label = iob[2:] if len(iob) > 2 else 'O'
                is_begin = (iob[0] == 'B')

                if label!=prev_label or is_begin:
                    if prev_label!='O':
                        entities[prev_label] = " ".join(entity)
                        entity = []

                entity.append(token)
                prev_label = label
            predicted_entities.append(entities)

        return predicted_entities