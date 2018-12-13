from pygears import gear, Intf
from pygears.typing import Int, Uint, Queue, Tuple, bitw
from pygears.common import fmap, czip, union_collapse, fifo
from pygears.common import ccat, cart, const, decoupler, mul, project
from pygears.cookbook import replicate, priority_mux

TDin = Queue[Uint['w_data']]
TCfg = Tuple[{'average_coef': Int['w_data'], 'average_window': Uint['w_data']}]


@gear
def accumulator(din, second_operand, delayed_din, *, w_data=16):
    return (din + second_operand - delayed_din) | Int[w_data]


@gear
def scale_input(din, *, shamt, w_data):
    return ((din | mul) >> shamt) | Int[w_data]


@gear
def moving_average(cfg: TCfg,
                   din: TDin,
                   *,
                   w_data=b'w_data',
                   shamt=15,
                   max_filter_ord=1024):

    din = din \
        | fmap(f=Int[16], lvl=1, fcat=czip)

    scaled_input = cart(cfg['average_coef'], din) \
        | fmap(f=scale_input(shamt=shamt, w_data=w_data),
               lvl=1,
               fcat=czip)

    second_operand = Intf(dtype=Int[w_data])

    din_window = scaled_input \
        | project \
        | fifo(depth=2**bitw(max_filter_ord))

    initial_load = ccat(cfg['average_window'], const(val=0, tout=Int[w_data])) \
        | replicate \
        | project

    delayed_din = (initial_load, din_window) \
        | priority_mux \
        | union_collapse

    average = scaled_input \
        | fmap(f=accumulator(second_operand,
                             delayed_din,
                             w_data=w_data),
               lvl=din.dtype.lvl,
               fcat=czip)

    average_reg = average \
        | project \
        | decoupler

    second_operand |= priority_mux(average_reg, const(val=0, tout=Int[w_data])) \
        | union_collapse

    return average
