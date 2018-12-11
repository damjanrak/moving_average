from pygears import gear, Intf
from pygears.typing import Int, Uint, Queue, Tuple
from pygears.common import fmap, add, union_collapse, fifo
from pygears.common import ccat, const, sub, decoupler
from pygears.cookbook import replicate, priority_mux

TDin = Queue[Int['w_data']]
TCfg = Tuple[{'average_coef': Int['w_data'], 'average_window': Uint['w_data']}]


@gear
def accumulator(din, second_operand, delayed_din, *, w_data=16):
    return (din + second_operand - delayed_din) | Uint[w_data]


@gear
def moving_average(cfg: TCfg, din: TDin, *, w_data=b'w_data'):

    second_operand = Intf(dtype=Int[w_data])

    din_window = din \
        | Int[w_data] \
        | fifo(depth=8)

    initial_load = ccat(cfg['average_window'], const(val=0, tout=Int[w_data])) \
        | replicate \
        | Int[w_data]

    delayed_din = (initial_load, din_window) \
        | priority_mux \
        | union_collapse

    accum = din \
        | fmap(f=accumulator(second_operand,
                             delayed_din,
                             w_data=w_data), lvl=din.dtype.lvl)

    accum_reg = accum | decoupler

    second_operand |= priority_mux(accum_reg | Int[w_data], const(val=0, tout=Int[w_data])) \
        | union_collapse

    return accum
