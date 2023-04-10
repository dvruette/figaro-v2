

import torch

import os
import glob
import hydra

import pytorch_lightning as pl

from models.seq2seq import Seq2SeqModule
from models.vae import VqVaeModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(ctx):
  ### Define available models ###

  available_models = [
    'vq-vae',
    'figaro-learned',
    'figaro-expert',
    'figaro',
    'figaro-inst',
    'figaro-chord',
    'figaro-meta',
    'figaro-no-inst',
    'figaro-no-chord',
    'figaro-no-meta',
    'baseline',
  ]
  
  if ctx.model.name not in available_models:
    raise ValueError(f"Unknown model: {ctx.model.name}")


  ### Create data loaders ###
  midi_files = glob.glob(os.path.join(ctx.data.input_dir, '**/*.mid'), recursive=True)
  if ctx.data.max_n_files > 0:
    midi_files = midi_files[:ctx.data.max_n_files]

  if len(midi_files) == 0:
    raise ValueError(f"WARNING: No MIDI files were found at '{ctx.data.input_dir}'. Did you download the dataset to the right location?")


  ### Create and train model ###

  # load model from checkpoint if available

  else:
    seq2seq_kwargs = {
      'encoder_layers': 4,
      'decoder_layers': 6,
      'num_attention_heads': 8,
      'intermediate_size': 2048,
      'd_model': D_MODEL,
      'context_size': MAX_CONTEXT,
      'lr': LEARNING_RATE,
      'warmup_steps': WARMUP_STEPS,
      'max_steps': MAX_STEPS,
    }
    dec_kwargs = { **seq2seq_kwargs }
    dec_kwargs['encoder_layers'] = 0

    # use lambda functions for lazy initialization
    model = {
      'vq-vae': lambda: VqVaeModule(
        encoder_layers=4,
        decoder_layers=6,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        n_codes=N_CODES, 
        n_groups=N_GROUPS, 
        context_size=MAX_CONTEXT,
        lr=LEARNING_RATE,
        lr_schedule=LR_SCHEDULE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        d_model=D_MODEL,
        d_latent=D_LATENT,
      ),
      'figaro-learned': lambda: Seq2SeqModule(
        description_flavor='latent',
        n_codes=vae_module.n_codes,
        n_groups=vae_module.n_groups,
        d_latent=vae_module.d_latent,
        **seq2seq_kwargs
      ),
      'figaro': lambda: Seq2SeqModule(
        description_flavor='both',
        n_codes=vae_module.n_codes,
        n_groups=vae_module.n_groups,
        d_latent=vae_module.d_latent,
        **seq2seq_kwargs
      ),
      'figaro-expert': lambda: Seq2SeqModule(
        description_flavor='description',
        **seq2seq_kwargs
      ),
      'figaro-no-meta': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': True, 'chords': True, 'meta': False },
        **seq2seq_kwargs
      ),
      'figaro-no-inst': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': False, 'chords': True, 'meta': True },
        **seq2seq_kwargs
      ),
      'figaro-no-chord': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': True, 'chords': False, 'meta': True },
        **seq2seq_kwargs
      ),
      'baseline': lambda: Seq2SeqModule(
        description_flavor='none',
        **dec_kwargs
      ),
    }[MODEL]()

  datamodule = model.get_datamodule(
    midi_files,
    vae_module=vae_module,
    batch_size=BATCH_SIZE, 
    num_workers=N_WORKERS, 
    pin_memory=True
  )

  checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor='valid_loss',
    dirpath=os.path.join(OUTPUT_DIR, MODEL),
    filename='{step}-{valid_loss:.2f}',
    save_last=True,
    save_top_k=2,
    every_n_train_steps=1000,
  )

  lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

  trainer = pl.Trainer(
    gpus=0 if device.type == 'cpu' else torch.cuda.device_count(),
    accelerator='dp',
    profiler='simple',
    callbacks=[checkpoint_callback, lr_monitor],
    max_epochs=EPOCHS,
    max_steps=MAX_TRAINING_STEPS,
    log_every_n_steps=max(100, min(25*ACCUMULATE_GRADS, 200)),
    val_check_interval=max(500, min(300*ACCUMULATE_GRADS, 1000)),
    limit_val_batches=64,
    auto_scale_batch_size=False,
    auto_lr_find=False,
    accumulate_grad_batches=ACCUMULATE_GRADS,
    stochastic_weight_avg=True,
    gradient_clip_val=1.0, 
    terminate_on_nan=True,
    resume_from_checkpoint=CHECKPOINT
  )

  trainer.fit(model, datamodule)

if __name__ == '__main__':
  main()