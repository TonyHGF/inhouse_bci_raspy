"""Plot channel-ablation delta R^2 and effective kernel distributions for five folds.

From the project directory with PYTHONPATH=src:
    python -m inhouse_bci_raspy.visualize.topomap --method both
Use --plot-only to regenerate figures from saved analysis arrays without inference.
"""

import argparse
import os
from pathlib import Path


def plot_grid(values, titles, channels, destination, title, label, limits, signed=False, columns=3):
    """Draw one comparable sensor topomap per row of values; save PNG and vector PDF.

    All panels share a caller-specified color range. Linear interpolation and a
    local convex-hull mask avoid cubic overshoot and unsupported scalp extrapolation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import mne
    import numpy as np

    info = mne.create_info(channels, sfreq=100, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='raise')
    if not np.isfinite(np.array([ch['loc'][:3] for ch in info['chs']])).all():
        raise ValueError('Missing sensor coordinates')
    rows = int(np.ceil(len(values) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.8*columns+.7, 3.65*rows+.5), squeeze=False)
    fig.subplots_adjust(left=.04, right=.86, bottom=.06, top=.86 if rows==1 else .9,
                        wspace=.22, hspace=.28)
    for i, ax in enumerate(axes.flat):
        if i >= len(values):
            ax.set_visible(False)
            continue
        im, _ = mne.viz.plot_topomap(values[i], info, axes=ax, show=False, names=channels,
            sensors=True, contours=0, image_interp='linear', extrapolate='local',
            sphere=(0.,0.,0.,.11), cmap='RdBu_r' if signed else 'viridis', vlim=limits)
        ax.set_title(titles[i], fontsize=11)
        for text in ax.texts:
            text.set_fontsize(7)
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white')])
    fig.suptitle(title, fontsize=14, y=.985)
    colorbar = fig.colorbar(im, cax=fig.add_axes([.90,.20,.018,.57]))
    colorbar.set_label(label)
    fig.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight', pad_inches=.15)
    fig.savefig(destination.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def render_all(destination: Path, metadata: dict) -> list:
    """Render saved numeric artifacts only, with globally consistent scales per metric family."""
    import numpy as np

    channels = metadata['channel_names']
    saved = np.load(destination / 'summary_arrays.npz')
    outputs, color_limits = [], {}

    def draw(name, values, titles, title, label, limits, signed=False, columns=3):
        plot_grid(values, titles, channels, destination/name, title, label, limits, signed, columns)
        outputs.extend([name+'.png', name+'.pdf'])
        color_limits[name] = list(limits)

    names = ['overall','left_hand','right_hand','both_hands','both_feet']
    if 'trial_fold_delta_r2' in saved:
        for level in ['trial','window']:
            folds, mean, sd = (saved[f'{level}_{part}_delta_r2'] for part in ['fold','mean','std'])
            bound = max(float(np.abs(folds).max()), float(np.abs(mean).max()), 1e-12)
            for k,name in enumerate(names):
                panels = np.concatenate([folds[:,:,k], mean[None,:,k]], axis=0)
                draw(f'ablation_{level}_{name}', panels,
                     [f'Fold {f}' for f in range(5)] + ['Weighted fold mean'],
                     f'Channel ablation | {level} | {name.replace("_"," ")}',
                     'Baseline R² minus ablated R²', (-bound,bound), signed=True)
            draw(f'ablation_{level}_std', sd.T, [n.replace('_',' ') for n in names],
                 f'Channel ablation | {level} | between-fold variability',
                 'Weighted population SD of delta R²', (0,max(float(sd.max()),1e-12)))
    if 'kernel_fold_fraction' in saved:
        folds, mean, sd = (100*saved[f'kernel_{part}_fraction'] for part in ['fold','mean','std'])
        draw('kernel_energy', np.concatenate([folds,mean[None]],axis=0),
             [f'Fold {f}' for f in range(5)] + ['Equal-weight fold mean'],
             'First-block kernel coefficient energy by input channel', 'Energy share (%)',
             (0,max(float(folds.max()),float(mean.max()),1e-12)))
        draw('kernel_energy_std', sd[None], ['Across five folds'],
             'Kernel energy share | between-fold variability', 'Population SD (percentage points)',
             (0,max(float(sd.max()),1e-12)),columns=1)
        kernels = [np.load(destination/f'fold_{f}'/'kernels.npz') for f in range(5)]
        bound = max(float(np.abs(k['folded_spatial']).max()) for k in kernels)
        bound = max(bound,1e-12)
        for fold,k in enumerate(kernels):
            titles = [f'Kernel {j} | temporal {int(parent)}' for j,parent in enumerate(k['parent_temporal_filter'])]
            draw(f'kernel_filters_fold_{fold}', k['folded_spatial'], titles,
                 f'Fold {fold} | signed spatial coefficients with BN slopes',
                 'Effective spatial coefficient × BN1 slope × BN2 slope', (-bound,bound), signed=True,columns=4)
        for k in kernels:
            k.close()
    saved.close()
    from ..io import save_json
    save_json(destination/'figure_manifest.json', {'files':outputs,'color_limits':color_limits,
        'interpolation':'linear','extrapolate':'local','montage':'standard_1020',
        'display_sphere_m':[0.,0.,0.,.11],
        'channel_order':channels})
    return outputs


def main() -> None:
    """CLI boundary: configure headless runtime, run analyses, then render their saved results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['both','ablation','kernel'],default='both')
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[3]/'outputs')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--overwrite',action='store_true',help='Recompute an existing analysis directory')
    parser.add_argument('--plot-only',action='store_true',help='Use saved analysis metadata/arrays; method flag is ignored')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    output = args.output_root.resolve()
    destination = output/'visualize/topomap'
    os.environ.setdefault('MPLCONFIGDIR',str(output/'matplotlib'))
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(output/'mne_home'))
    from ..analysis.channel_pipeline import run_analysis
    from ..analysis.channel_inputs import read_json
    from ..io import save_json
    from .topomap_report import write_report
    if args.plot_only:
        metadata = read_json(destination/'metadata.json')
    else:
        metadata = run_analysis(output,destination,args.method,args.device,args.batch_size,args.overwrite)
    try:
        save_json(destination/'status.json',{'status':'rendering','method':metadata['method'],'folds_complete':5})
        print('Rendering comparable sensor maps...',flush=True)
        figures = render_all(destination,metadata)
        write_report(destination,metadata,figures)
        save_json(destination/'status.json',{'status':'complete','method':metadata['method'],
            'folds_complete':5,'figure_files':len(figures),'report':str(destination/'report.md')})
        print(f'Complete: {destination / "report.md"}',flush=True)
    except Exception as error:
        save_json(destination/'status.json',{'status':'render_failed','error':str(error)})
        raise


if __name__ == '__main__':
    main()
