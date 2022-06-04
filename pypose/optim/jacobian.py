import sys
import math
import copy
import torch
import warnings
import functorch
from torch import nn, Tensor
from torch.autograd.functional import jacobian
from typing import List, Tuple, Dict, Union, Callable


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return names, params


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


def modjac(model, inputs, flatten=False, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode'):
    r'''
    Compute the model Jacobian with respect to the model parameters.

    Args:
        model (torch.nn.Module): a PyTorch model that takes Tensor inputs and
            returns a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        flatten (bool, optional): If ``True``, all module parameters are flattened
            and concatenated to form a single vector. The Jacobian will be computed
            with respect to this single flattened parameter.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        output and ``flatten=True``, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If either one doesn't
        hold, then the Jacobian will be a tuple of Tensors. If both of
        them don't hold, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th parameter and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding parameter and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the parameter.

    Warning:
        The function :obj:`modjac` calculate Jacobian of model parameters.
        This is in contrast to PyTorch's function `jacobian
        <https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html>`_,
        which computes the Jacobian of a given Python function.

    Example:

        Calculates Jacobian with respect to all model parameters.

        >>> model = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        >>> inputs = torch.randn(1, 1, 1)
        >>> J = pp.optim.modjac(model, inputs)
        (tensor([[[[[[[0.3365]]]]]]]), tensor([[[[1.]]]]))
        >>> [j.shape for j in J]
        [torch.Size([1, 1, 1, 1, 1, 1, 1]), torch.Size([1, 1, 1, 1])]

        Function with flattened parameters returns a combined Jacobian.

        >>> inputs = torch.randn(2, 2, 2)
        >>> model = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        >>> J = pp.optim.modjac(model, inputs, flatten=True)
        tensor([[[[-1.1571, -1.6217,  0.0000,  0.0000,  1.0000,  0.0000],
                  [ 0.2917, -1.1545,  0.0000,  0.0000,  1.0000,  0.0000]],
                 [[-1.4052,  0.7642,  0.0000,  0.0000,  1.0000,  0.0000],
                  [ 0.7777, -1.5251,  0.0000,  0.0000,  1.0000,  0.0000]]],
                [[[ 0.0000,  0.0000, -1.1571, -1.6217,  0.0000,  1.0000],
                  [ 0.0000,  0.0000,  0.2917, -1.1545,  0.0000,  1.0000]],
                 [[ 0.0000,  0.0000, -1.4052,  0.7642,  0.0000,  1.0000],
                  [ 0.0000,  0.0000,  0.7777, -1.5251,  0.0000,  1.0000]]]])
        >>> J.shape
        torch.Size([2, 2, 2, 6])

        Calculate Jacobian with respect to :obj:`pypose.LieTensor`.

        >>> class PoseTransform(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.p = pp.Parameter(pp.randn_so3(2))
        ...
        ...     def forward(self, x):
        ...         return self.p.Exp() * x
        ...
        >>> model, inputs = PoseTransform(), pp.randn_SO3()
        >>> J = pp.optim.modjac(model, inputs, flatten=True)
        tensor([[[ 0.6769,  0.4854,  0.3703,  0.0000,  0.0000,  0.0000],
                 [-0.6020,  0.6618,  0.1506,  0.0000,  0.0000,  0.0000],
                 [-0.1017, -0.3866,  0.8824,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
                [[ 0.0000,  0.0000,  0.0000,  0.9671,  0.1685, -0.1407],
                 [ 0.0000,  0.0000,  0.0000, -0.2092,  0.9203, -0.2629],
                 [ 0.0000,  0.0000,  0.0000,  0.0665,  0.2907,  0.9380],
                 [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]])
    '''
    names, params = extract_weights(model) # deparameterize weights

    if flatten is True:
        numels, shapes, params = zip(*[(p.numel(), p.shape, p.view(-1)) for p in params])
        params = torch.cat(params, dim=-1)

    def param_as_input(*params):
        if flatten is True:
            params = torch.split(*params, numels)
            params = [p.view(s) for p, s in zip(params, shapes)]
        load_weights(model, names, params)
        return model(inputs)

    return jacobian(param_as_input, params, create_graph, strict, vectorize, strategy)


def modjacrev(model, inputs, argnums=0, *, has_aux=False):
    func, params = functorch.make_functional(model)
    jacrev = functorch.jacrev(func, argnums=argnums, has_aux=has_aux)
    return jacrev(params, inputs)


def modjacfwd(model, inputs, argnums=0, *, has_aux=False):
    func, params = functorch.make_functional(model)
    jacfwd = functorch.jacfwd(func, argnums=argnums, has_aux=has_aux)
    return jacfwd(params, inputs)