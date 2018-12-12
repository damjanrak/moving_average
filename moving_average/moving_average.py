from pygears import gear, Intf
from pygears.typing import Int, Uint, Queue, Tuple
from pygears.common import fmap, czip, union_collapse, fifo
from pygears.common import ccat, cart, const, decoupler, mul, shr
from pygears.cookbook import replicate, priority_mux

TDin = Queue[Uint['w_data']]
TCfg = Tuple[{'average_coef': Int['w_data'], 'average_window': Uint['w_data']}]


@gear
def accumulator(din, second_operand, delayed_din, *, w_data=16):
    return (din + second_operand - delayed_din) | Int[w_data]


@gear
def shift(din):
    return (din >> 15) | Int[16]


@gear
def moving_average(cfg: TCfg, din: TDin, *, w_data=b'w_data', shamt=15):

    print(f'tipic {din.dtype}')
    din = din \
        | fmap(f=Int[16], lvl=1, fcat=czip)

    print(f'tipic {din.dtype}')
    scaled_input = cart(cfg['average_coef'], din) \
        | fmap(f=mul, lvl=din.dtype.lvl, fcat=czip)

    scaled_shr = scaled_input \
        | fmap(f=shift, lvl=1, fcat=czip)

    second_operand = Intf(dtype=Int[w_data])

    din_window = scaled_shr \
        | Int[w_data] \
        | fifo(depth=128)

    initial_load = ccat(cfg['average_window'], const(val=0, tout=Int[w_data])) \
        | replicate \
        | Int[w_data]

    delayed_din = (initial_load, din_window) \
        | priority_mux \
        | union_collapse

    accum = scaled_shr \
        | fmap(f=accumulator(second_operand,
                             delayed_din,
                             w_data=w_data), lvl=din.dtype.lvl, fcat=czip)

    accum_reg = accum | decoupler

    second_operand |= priority_mux(accum_reg | Int[w_data], const(val=0, tout=Int[w_data])) \
        | union_collapse

    return accum
