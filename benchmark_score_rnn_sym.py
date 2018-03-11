import mxnet as mx
import time
import logging

#[bs, sequence length, embedding size, hidden size]
input_shape_list = [[64,15,500,500],
   [64,20,500,500],
   [64,25,500,500],
   [64,30,500,500],
   [64,35,500,500],
   [64,40,500,500],
   [64,45,500,500],
   [64,50,500,500],
   [16,25,512,512],
   [32,25,512,512],
   [64,25,512,512],
   [128,25,512,512],
   [16,25,1024,1024],
   [32,25,1024,1024],
   [64,25,1024,1024],
   [128,25,1024,1024],
   [16,25,2048,2048],
   [32,25,2048,2048],
   [64,25,2048,2048],
   [128,25,2048,2048],
   [16,25,4096,4096],
   [32,25,4096,4096],
   [64,25,4096,4096],
   [128,25,4096,4096]]

rnncell_type = ['rnn', 'lstm', 'gru', 'sru']

dry_run = 10
iter_num = 100

def rnncell_score_stacked(input_data, cell_type, ctx, layout='NTC'):
    if cell_type not in rnncell_type:
        assert False

    bs = input_data[0]
    seq_len = input_data[1]
    embed_dim = input_data[2]
    hidden_size = input_data[3]
    if layout == 'NTC':
        dshape = (bs, seq_len, embed_dim)
    elif layout == 'TNC':
        logging.warning('layout TNC is used!')
        dshape = (seq_len, bs, embed_dim) 
    data = mx.sym.Variable('data')

    tic = time.time()
    #default layout='NTC'
    if cell_type == 'lstm':
        lstm_cell = mx.rnn.LSTMCell(hidden_size, prefix='l0_')
        output, (HY, CY) = lstm_cell.unroll(seq_len, data, layout=layout, merge_outputs=True)
        #group = mx.symbol.Group([output, HY, CY])
    elif cell_type == 'gru':
        gru_cell = mx.rnn.GRUCell(hidden_size, prefix='l0_')
        output, _ = gru_cell.unroll(seq_len, data, layout=layout, merge_outputs=True)
    #elif cell_type == 'rnn':

    #elif cell_type == 'sru':

    mod = mx.mod.Module(output, label_names=None, context=ctx)
    mod.bind(data_shapes=[('data', dshape)], label_shapes=None)
    batch = mx.io.DataBatch(data=[mx.random.uniform(shape=dshape)], label=[])
    mod.init_params()

    mod.forward(batch, is_train=False)
   
    for output in mod.get_outputs():
        output.wait_to_read()

    fwd = time.time() - tic

    return fwd


def rnncell_score_fused(input_data, cell_type, ctx, layout='NTC'):
    if cell_type not in rnncell_type:
        assert False

    bs = input_data[0]
    seq_len = input_data[1]
    embed_dim = input_data[2]
    hidden_size = input_data[3]
    if layout == 'NTC':
        dshape = (bs, seq_len, embed_dim)
    elif layout == 'TNC':
        #logging.warning('layout TNC is used!')
        dshape = (seq_len, bs, embed_dim)
    data = mx.sym.Variable('data')

    tic = time.time()
    #default layout = 'NTC'
    if cell_type == 'lstm':
        lstm_cell = mx.rnn.FusedRNNCell(hidden_size, mode='lstm', get_next_state=False, prefix='l0_')
        rnn_sym, _ = lstm_cell.unroll(seq_len, data, layout=layout, merge_outputs=True)
    elif cell_type == 'gru':
        gru_cell = mx.rnn.FusedRNNCell(hidden_size, mode='gru', prefix='l0_')
        rnn_sym, _ = gru_cell.unroll(seq_len, data, layout=layout, merge_outputs=True)
    #elif cell_type == 'rnn':

    #elif cell_type == 'sru':

    mod = mx.mod.Module(rnn_sym, label_names=None, context=ctx)
    mod.bind(data_shapes=[('data', dshape)], label_shapes=None)
    batch = mx.io.DataBatch(data=[mx.random.uniform(shape=dshape)], label=[])
    mod.init_params()
    
    mod.forward(batch, is_train=False)
    output = mod.get_outputs()[0]
    output.wait_to_read()
    fwd = time.time() - tic
  
    return fwd


if __name__ == '__main__':

    total_fwd = 0
    for input_shape in input_shape_list:
        for i in range(dry_run + iter_num):
            fwd = rnncell_score_fused(input_shape, 'lstm', mx.cpu(), layout='TNC')
            if i >= dry_run:
                total_fwd += fwd
        total_fwd = total_fwd/iter_num
        print(str(input_shape) + ' cost ' + str(total_fwd) + 's SPS = '+ str(input_shape[0]/total_fwd))
    
    total_fwd = 0
    for input_shape in input_shape_list:
        for i in range(dry_run + iter_num):
            fwd = rnncell_score_stacked(input_shape, 'lstm', mx.cpu())
            if i >= dry_run:
                total_fwd += fwd
        total_fwd = total_fwd/iter_num  
        print(str(input_shape) + ' cost ' + str(total_fwd) + 's SPS = '+ str(input_shape[0]/total_fwd))
    









