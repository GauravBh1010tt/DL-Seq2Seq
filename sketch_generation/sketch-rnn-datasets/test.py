# -*- coding: utf-8 -*-

idx = 219
draw_image(data_enc[idx][:,[0,1,3]])
inp = torch.tensor(data_enc[idx:idx+1], dtype=torch.float, device=device)
strokes, mix_params = skrnn_sample(encoder, decoder, hidden_dec_dim, latent_dim, inp_enc = inp,random_state=98,
                                       time_step=max_seq_len, cond_gen=cond_gen,bi_mode= bi_mode, device=device)
draw_image(strokes)