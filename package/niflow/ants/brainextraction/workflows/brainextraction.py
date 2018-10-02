#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Nipype translation of ANTs workflows
------------------------------------

"""
from __future__ import print_function, division, absolute_import, unicode_literals

# general purpose
import os
from multiprocessing import cpu_count
from pkg_resources import resource_filename as pkgr_fn
from packaging.version import parse as parseversion, Version
from pathlib import Path

# nipype
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.ants import N4BiasFieldCorrection

# niworkflows
from ..interfaces.ants import (
    ImageMath,
    ResampleImageBySpacing,
    AI,
    ThresholdImage,
)
from ..interfaces.fixes import (
    FixHeaderRegistration as Registration,
    FixHeaderApplyTransforms as ApplyTransforms,
)

from .atropos import ATROPOS_MODELS, init_atropos_wf


def init_brain_extraction_wf(name='brain_extraction_wf',
                             in_template='OASIS',
                             use_float=True,
                             normalization_quality='precise',
                             omp_nthreads=None,
                             mem_gb=3.0,
                             modality='T1',
                             atropos_refine=True,
                             atropos_use_random_seed=True,
                             atropos_model=None):
    """
    A Nipype implementation of the official ANTs' ``antsBrainExtraction.sh``
    workflow (only for 3D images).

    The official workflow is built as follows (and this implementation
    follows the same organization):

      1. Step 1 performs several clerical tasks (adding padding, calculating
         the Laplacian of inputs, affine initialization) and the core
         spatial normalization.
      2. Maps the brain mask into target space using the normalization
         calculated in 1.
      3. Superstep 1b: smart binarization of the brain mask
      4. Superstep 6: apply ATROPOS and massage its outputs
      5. Superstep 7: use results from 4 to refine the brain mask


    .. workflow::
        :graph2use: orig
        :simple_form: yes
        from niworkflows.anat import init_brain_extraction_wf
        wf = init_brain_extraction_wf()


    **Parameters**

        in_template : str
            Name of the skull-stripping template ('OASIS', 'NKI', or
            path).
            The brain template from which regions will be projected
            Anatomical template created using e.g. LPBA40 data set with
            ``buildtemplateparallel.sh`` in ANTs.
            The workflow will automatically search for a brain probability
            mask created using e.g. LPBA40 data set which have brain masks
            defined, and warped to anatomical template and
            averaged resulting in a probability image.
        use_float : bool
            Whether single precision should be used
        normalization_quality : str
            Use more precise or faster registration parameters
            (default: ``precise``, other possible values: ``testing``)
        omp_nthreads : int
            Maximum number of threads an individual process may use
        mem_gb : float
            Estimated peak memory consumption of the most hungry nodes
            in the workflow
        modality : str
            Sequence type of the first input image ('T1', 'T2', or 'FLAIR')
        atropos_refine : bool
            Enables or disables the whole ATROPOS sub-workflow
        atropos_use_random_seed : bool
            Whether ATROPOS should generate a random seed based on the
            system's clock
        atropos_model : tuple or None
            Allows to specify a particular segmentation model, overwriting
            the defaults based on ``modality``
        name : str, optional
            Workflow name (default: antsBrainExtraction)


    **Inputs**

        in_files
            List of input anatomical images to be brain-extracted,
            typically T1-weighted.
            If a list of anatomical images is provided, subsequently
            specified images are used during the segmentation process.
            However, only the first image is used in the registration
            of priors.
            Our suggestion would be to specify the T1w as the first image.
        in_mask
            (optional) Mask used for registration to limit the metric
            computation to a specific region.


    **Outputs**

        bias_corrected
            The ``in_files`` input images, after :abbr:`INU (intensity non-uniformity)`
            correction.
        out_mask
            Calculated brain mask
        bias_image
            The :abbr:`INU (intensity non-uniformity)` field estimated for each
            input in ``in_files``
        out_segm
            Output segmentation by ATROPOS
        out_tpms
            Output :abbr:`TPMs (tissue probability maps)` by ATROPOS


    """
    wf = pe.Workflow(name)

    template_path = None
    if in_template == 'OASIS':
        template_path = pkgr_fn('niflow.ants.brainextraction',
                                'data/tpl-OASIS30ANTs')
    else:
        template_path = in_template

    mod = ('%sw' % modality[:2].upper()
           if modality.upper().startswith('T') else modality.upper())

    # Append template modality
    potential_targets = list(Path(template_path).glob('*_%s.nii.gz' % mod))
    if not potential_targets:
        raise ValueError(
            'No %s template was found under "%s".' % (mod, template_path))

    tpl_target_path = str(potential_targets[0])
    target_basename = '_'.join(tpl_target_path.split('_')[:-1])

    # Get probabilistic brain mask if available
    tpl_mask_path = '%s_class-brainmask_probtissue.nii.gz' % target_basename
    # Fall-back to a binary mask just in case
    if not os.path.exists(tpl_mask_path):
        tpl_mask_path = '%s_brainmask.nii.gz' % target_basename

    if not os.path.exists(tpl_mask_path):
        raise ValueError(
            'Probability map for the brain mask associated to this template '
            '"%s" not found.' % tpl_mask_path)

    if omp_nthreads is None or omp_nthreads < 1:
        omp_nthreads = cpu_count()

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_files', 'in_mask']),
                        name='inputnode')

    # Try to find a registration mask, set if available
    tpl_regmask_path = '%s_label-BrainCerebellumRegistration_roi.nii.gz' % target_basename
    if os.path.exists(tpl_regmask_path):
        inputnode.inputs.in_mask = tpl_regmask_path

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bias_corrected', 'out_mask', 'bias_image', 'out_segm']),
        name='outputnode')

    trunc = pe.MapNode(ImageMath(operation='TruncateImageIntensity', op2='0.01 0.999 256'),
                       name='truncate_images', iterfield=['op1'])
    inu_n4 = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3, save_bias=True, copy_header=True,
            n_iterations=[50] * 4, convergence_threshold=1e-7, shrink_factor=4,
            bspline_fitting_distance=200),
        n_procs=omp_nthreads, name='inu_n4', iterfield=['input_image'])

    res_tmpl = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                       apply_smoothing=True), name='res_tmpl')
    res_tmpl.inputs.input_image = tpl_target_path
    res_target = pe.Node(ResampleImageBySpacing(out_spacing=(4, 4, 4),
                         apply_smoothing=True), name='res_target')

    lap_tmpl = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                       name='lap_tmpl')
    lap_tmpl.inputs.op1 = tpl_target_path
    lap_target = pe.Node(ImageMath(operation='Laplacian', op2='1.5 1'),
                         name='lap_target')
    mrg_tmpl = pe.Node(niu.Merge(2), name='mrg_tmpl')
    mrg_tmpl.inputs.in1 = tpl_target_path
    mrg_target = pe.Node(niu.Merge(2), name='mrg_target')

    # Initialize transforms with antsAI
    init_aff = pe.Node(AI(
        metric=('Mattes', 32, 'Regular', 0.2),
        transform=('Affine', 0.1),
        search_factor=(20, 0.12),
        principal_axes=False,
        convergence=(10, 1e-6, 10),
        verbose=True),
        name='init_aff',
        n_procs=omp_nthreads)

    if parseversion(Registration().version) > Version('2.2.0'):
        init_aff.inputs.search_grid = (40, (0, 40, 40))

    # Set up spatial normalization
    norm = pe.Node(Registration(
        from_file=pkgr_fn(
            'niflow.ants.brainextraction',
            'data/antsBrainExtraction_%s.json' % normalization_quality)),
        name='norm',
        n_procs=omp_nthreads,
        mem_gb=mem_gb)
    norm.inputs.float = use_float
    fixed_mask_trait = 'fixed_image_mask'
    if parseversion(Registration().version) >= Version('2.2.0'):
        fixed_mask_trait += 's'

    map_brainmask = pe.Node(
        ApplyTransforms(interpolation='Gaussian', float=True),
        name='map_brainmask',
        mem_gb=1
    )
    map_brainmask.inputs.input_image = tpl_mask_path

    thr_brainmask = pe.Node(ThresholdImage(
        dimension=3, th_low=0.5, th_high=1.0, inside_value=1,
        outside_value=0), name='thr_brainmask')

    # Morphological dilation, radius=2
    dil_brainmask = pe.Node(ImageMath(operation='MD', op2='2'),
                            name='dil_brainmask')
    # Get largest connected component
    get_brainmask = pe.Node(ImageMath(operation='GetLargestComponent'),
                            name='get_brainmask')

    # Apply mask
    apply_mask = pe.MapNode(ApplyMask(), iterfield=['in_file'], name='apply_mask')

    wf.connect([
        (inputnode, trunc, [('in_files', 'op1')]),
        (inputnode, init_aff, [('in_mask', 'fixed_image_mask')]),
        (inputnode, norm, [('in_mask', fixed_mask_trait)]),
        (inputnode, map_brainmask, [(('in_files', _pop), 'reference_image')]),
        (trunc, inu_n4, [('output_image', 'input_image')]),
        (inu_n4, res_target, [
            (('output_image', _pop), 'input_image')]),
        (inu_n4, lap_target, [
            (('output_image', _pop), 'op1')]),
        (res_tmpl, init_aff, [('output_image', 'fixed_image')]),
        (res_target, init_aff, [('output_image', 'moving_image')]),
        (inu_n4, mrg_target, [('output_image', 'in1')]),
        (lap_tmpl, mrg_tmpl, [('output_image', 'in2')]),
        (lap_target, mrg_target, [('output_image', 'in2')]),

        (init_aff, norm, [('output_transform', 'initial_moving_transform')]),
        (mrg_tmpl, norm, [('out', 'fixed_image')]),
        (mrg_target, norm, [('out', 'moving_image')]),
        (norm, map_brainmask, [
            ('reverse_invert_flags', 'invert_transform_flags'),
            ('reverse_transforms', 'transforms')]),
        (map_brainmask, thr_brainmask, [('output_image', 'input_image')]),
        (thr_brainmask, dil_brainmask, [('output_image', 'op1')]),
        (dil_brainmask, get_brainmask, [('output_image', 'op1')]),
        (inu_n4, apply_mask, [('output_image', 'in_file')]),
        (get_brainmask, apply_mask, [('output_image', 'mask_file')]),
        (get_brainmask, outputnode, [('output_image', 'out_mask')]),
        (apply_mask, outputnode, [('out_file', 'bias_corrected')]),
        (inu_n4, outputnode, [('bias_image', 'bias_image')]),
    ])

    if atropos_refine:
        atropos_wf = init_atropos_wf(
            use_random_seed=atropos_use_random_seed,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            in_segmentation_model=atropos_model or list(ATROPOS_MODELS[modality].values())
        )

        wf.disconnect([
            (get_brainmask, outputnode, [('output_image', 'out_mask')]),
            (get_brainmask, apply_mask, [('output_image', 'mask_file')]),
        ])
        wf.connect([
            (inu_n4, atropos_wf, [
                ('output_image', 'inputnode.in_files')]),
            (get_brainmask, atropos_wf, [
                ('output_image', 'inputnode.in_mask')]),
            (atropos_wf, outputnode, [
                ('outputnode.out_mask', 'out_mask')]),
            (atropos_wf, apply_mask, [
                ('outputnode.out_mask', 'mask_file')]),
            (atropos_wf, outputnode, [
                ('outputnode.out_segm', 'out_segm'),
                ('outputnode.out_tpms', 'out_tpms')])
        ])
    return wf


def _pop(in_files):
    if isinstance(in_files, (list, tuple)):
        return in_files[0]
    return in_files
