from collections import OrderedDict

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from nipype.interfaces.ants import Atropos, MultiplyImages

from ..interfaces.ants import ImageMath

ATROPOS_MODELS = {
    'T1': OrderedDict([
        ('nclasses', 3),
        ('csf', 1),
        ('gm', 2),
        ('wm', 3),
    ]),
    'T2': OrderedDict([
        ('nclasses', 3),
        ('csf', 3),
        ('gm', 2),
        ('wm', 1),
    ]),
    'FLAIR': OrderedDict([
        ('nclasses', 3),
        ('csf', 1),
        ('gm', 3),
        ('wm', 2),
    ]),
}


def init_atropos_wf(name='atropos_wf',
                    use_random_seed=True,
                    omp_nthreads=None,
                    mem_gb=3.0,
                    padding=10,
                    in_segmentation_model=list(ATROPOS_MODELS['T1'].values())):
    """
    Implements supersteps 6 and 7 of ``antsBrainExtraction.sh``,
    which refine the mask previously computed with the spatial
    normalization to the template.

    **Parameters**

        use_random_seed : bool
            Whether ATROPOS should generate a random seed based on the
            system's clock
        omp_nthreads : int
            Maximum number of threads an individual process may use
        mem_gb : float
            Estimated peak memory consumption of the most hungry nodes
            in the workflow
        padding : int
            Pad images with zeros before processing
        in_segmentation_model : tuple
            A k-means segmentation is run to find gray or white matter
            around the edge of the initial brain mask warped from the
            template.
            This produces a segmentation image with :math:`$K$` classes,
            ordered by mean intensity in increasing order.
            With this option, you can control  :math:`$K$` and tell
            the script which classes represent CSF, gray and white matter.
            Format (K, csfLabel, gmLabel, wmLabel).
            Examples:
              - ``(3,1,2,3)`` for T1 with K=3, CSF=1, GM=2, WM=3 (default)
              - ``(3,3,2,1)`` for T2 with K=3, CSF=3, GM=2, WM=1
              - ``(3,1,3,2)`` for FLAIR with K=3, CSF=1 GM=3, WM=2
              - ``(4,4,2,3)`` uses K=4, CSF=4, GM=2, WM=3
        name : str, optional
            Workflow name (default: atropos_wf)


    **Inputs**

        in_files
            :abbr:`INU (intensity non-uniformity)`-corrected files.
        in_mask
            Brain mask calculated previously


    **Outputs**
        out_mask
            Refined brain mask
        out_segm
            Output segmentation
        out_tpms
            Output :abbr:`TPMs (tissue probability maps)`

    """
    wf = pe.Workflow(name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_files', 'in_mask']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_mask', 'out_segm', 'out_tpms']), name='outputnode')

    # Run atropos (core node)
    atropos = pe.Node(Atropos(
        dimension=3,
        initialization='KMeans',
        number_of_tissue_classes=in_segmentation_model[0],
        n_iterations=3,
        convergence_threshold=0.0,
        mrf_radius=[1, 1, 1],
        mrf_smoothing_factor=0.1,
        likelihood_model='Gaussian',
        use_random_seed=use_random_seed),
        name='01_atropos', n_procs=omp_nthreads, mem_gb=mem_gb)

    # massage outputs
    pad_segm = pe.Node(ImageMath(operation='PadImage', op2='%d' % padding),
                       name='02_pad_segm')
    pad_mask = pe.Node(ImageMath(operation='PadImage', op2='%d' % padding),
                       name='03_pad_mask')

    # Split segmentation in binary masks
    sel_labels = pe.Node(niu.Function(function=_select_labels,
                         output_names=['out_wm', 'out_gm', 'out_csf']),
                         name='04_sel_labels')
    sel_labels.inputs.labels = list(reversed(in_segmentation_model[1:]))

    # Select largest components (GM, WM)
    # ImageMath ${DIMENSION} ${EXTRACTION_WM} GetLargestComponent ${EXTRACTION_WM}
    get_wm = pe.Node(ImageMath(operation='GetLargestComponent'),
                     name='05_get_wm')
    get_gm = pe.Node(ImageMath(operation='GetLargestComponent'),
                     name='06_get_gm')

    # Fill holes and calculate intersection
    # ImageMath ${DIMENSION} ${EXTRACTION_TMP} FillHoles ${EXTRACTION_GM} 2
    # MultiplyImages ${DIMENSION} ${EXTRACTION_GM} ${EXTRACTION_TMP} ${EXTRACTION_GM}
    fill_gm = pe.Node(ImageMath(operation='FillHoles', op2='2'),
                      name='07_fill_gm')
    mult_gm = pe.Node(MultiplyImages(dimension=3), name='08_mult_gm')

    # MultiplyImages ${DIMENSION} ${EXTRACTION_WM} ${ATROPOS_WM_CLASS_LABEL} ${EXTRACTION_WM}
    # ImageMath ${DIMENSION} ${EXTRACTION_TMP} ME ${EXTRACTION_CSF} 10
    relabel_wm = pe.Node(MultiplyImages(dimension=3, second_input=in_segmentation_model[-1]),
                         name='09_relabel_wm')
    me_csf = pe.Node(ImageMath(operation='ME', op2='10'), name='10_me_csf')

    # ImageMath ${DIMENSION} ${EXTRACTION_GM} addtozero ${EXTRACTION_GM} ${EXTRACTION_TMP}
    # MultiplyImages ${DIMENSION} ${EXTRACTION_GM} ${ATROPOS_GM_CLASS_LABEL} ${EXTRACTION_GM}
    # ImageMath ${DIMENSION} ${EXTRACTION_SEGMENTATION} addtozero ${EXTRACTION_WM} ${EXTRACTION_GM}
    add_gm = pe.Node(ImageMath(operation='addtozero'),
                     name='11_add_gm')
    relabel_gm = pe.Node(MultiplyImages(dimension=3, second_input=in_segmentation_model[-2]),
                         name='12_relabel_gm')
    add_gm_wm = pe.Node(ImageMath(operation='addtozero'),
                        name='13_add_gm_wm')

    # Superstep 7
    # Split segmentation in binary masks
    sel_labels2 = pe.Node(niu.Function(function=_select_labels,
                          output_names=['out_wm', 'out_gm', 'out_csf']),
                          name='14_sel_labels2')
    sel_labels2.inputs.labels = list(reversed(in_segmentation_model[1:]))

    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} addtozero ${EXTRACTION_MASK} ${EXTRACTION_TMP}
    add_7 = pe.Node(ImageMath(operation='addtozero'), name='15_add_7')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} ME ${EXTRACTION_MASK} 2
    me_7 = pe.Node(ImageMath(operation='ME', op2='2'), name='16_me_7')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} GetLargestComponent ${EXTRACTION_MASK}
    comp_7 = pe.Node(ImageMath(operation='GetLargestComponent'),
                     name='17_comp_7')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} MD ${EXTRACTION_MASK} 4
    md_7 = pe.Node(ImageMath(operation='MD', op2='4'), name='18_md_7')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} FillHoles ${EXTRACTION_MASK} 2
    fill_7 = pe.Node(ImageMath(operation='FillHoles', op2='2'),
                     name='19_fill_7')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} addtozero ${EXTRACTION_MASK} \
    # ${EXTRACTION_MASK_PRIOR_WARPED}
    add_7_2 = pe.Node(ImageMath(operation='addtozero'), name='20_add_7_2')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} MD ${EXTRACTION_MASK} 5
    md_7_2 = pe.Node(ImageMath(operation='MD', op2='5'), name='21_md_7_2')
    # ImageMath ${DIMENSION} ${EXTRACTION_MASK} ME ${EXTRACTION_MASK} 5
    me_7_2 = pe.Node(ImageMath(operation='ME', op2='5'), name='22_me_7_2')

    # De-pad
    depad_mask = pe.Node(ImageMath(operation='PadImage', op2='-%d' % padding),
                         name='23_depad_mask')
    depad_segm = pe.Node(ImageMath(operation='PadImage', op2='-%d' % padding),
                         name='24_depad_segm')
    depad_gm = pe.Node(ImageMath(operation='PadImage', op2='-%d' % padding),
                       name='25_depad_gm')
    depad_wm = pe.Node(ImageMath(operation='PadImage', op2='-%d' % padding),
                       name='26_depad_wm')
    depad_csf = pe.Node(ImageMath(operation='PadImage', op2='-%d' % padding),
                        name='27_depad_csf')

    merge_tpms = pe.Node(niu.Merge(in_segmentation_model[0]), name='merge_tpms')
    wf.connect([
        (inputnode, pad_mask, [('in_mask', 'op1')]),
        (inputnode, atropos, [('in_files', 'intensity_images'),
                              ('in_mask', 'mask_image')]),
        (atropos, pad_segm, [('classified_image', 'op1')]),
        (pad_segm, sel_labels, [('output_image', 'in_segm')]),
        (sel_labels, get_wm, [('out_wm', 'op1')]),
        (sel_labels, get_gm, [('out_gm', 'op1')]),
        (get_gm, fill_gm, [('output_image', 'op1')]),
        (get_gm, mult_gm, [('output_image', 'first_input'),
                           (('output_image', _gen_name), 'output_product_image')]),
        (fill_gm, mult_gm, [('output_image', 'second_input')]),
        (get_wm, relabel_wm, [('output_image', 'first_input'),
                              (('output_image', _gen_name), 'output_product_image')]),
        (sel_labels, me_csf, [('out_csf', 'op1')]),
        (mult_gm, add_gm, [('output_product_image', 'op1')]),
        (me_csf, add_gm, [('output_image', 'op2')]),
        (add_gm, relabel_gm, [('output_image', 'first_input'),
                              (('output_image', _gen_name), 'output_product_image')]),
        (relabel_wm, add_gm_wm, [('output_product_image', 'op1')]),
        (relabel_gm, add_gm_wm, [('output_product_image', 'op2')]),
        (add_gm_wm, sel_labels2, [('output_image', 'in_segm')]),
        (sel_labels2, add_7, [('out_wm', 'op1'),
                              ('out_gm', 'op2')]),
        (add_7, me_7, [('output_image', 'op1')]),
        (me_7, comp_7, [('output_image', 'op1')]),
        (comp_7, md_7, [('output_image', 'op1')]),
        (md_7, fill_7, [('output_image', 'op1')]),
        (fill_7, add_7_2, [('output_image', 'op1')]),
        (pad_mask, add_7_2, [('output_image', 'op2')]),
        (add_7_2, md_7_2, [('output_image', 'op1')]),
        (md_7_2, me_7_2, [('output_image', 'op1')]),
        (me_7_2, depad_mask, [('output_image', 'op1')]),
        (add_gm_wm, depad_segm, [('output_image', 'op1')]),
        (relabel_wm, depad_wm, [('output_product_image', 'op1')]),
        (relabel_gm, depad_gm, [('output_product_image', 'op1')]),
        (sel_labels, depad_csf, [('out_csf', 'op1')]),
        (depad_csf, merge_tpms, [('output_image', 'in1')]),
        (depad_gm, merge_tpms, [('output_image', 'in2')]),
        (depad_wm, merge_tpms, [('output_image', 'in3')]),
        (depad_mask, outputnode, [('output_image', 'out_mask')]),
        (depad_segm, outputnode, [('output_image', 'out_segm')]),
        (merge_tpms, outputnode, [('out', 'out_tpms')]),
    ])
    return wf


def _select_labels(in_segm, labels):
    from os import getcwd
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_files = []

    cwd = getcwd()
    nii = nb.load(in_segm)
    for l in labels:
        data = (nii.get_data() == l).astype(np.uint8)
        newnii = nii.__class__(data, nii.affine, nii.header)
        newnii.set_data_dtype('uint8')
        out_file = fname_presuffix(in_segm, suffix='class-%02d' % l,
                                   newpath=cwd)
        newnii.to_filename(out_file)
        out_files.append(out_file)
    return out_files


def _gen_name(in_file):
    import os
    from nipype.utils.filemanip import fname_presuffix
    return os.path.basename(fname_presuffix(in_file, suffix='processed'))
