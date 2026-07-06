clear;clc
rootD = '/isilon/LFMI/VMdrive/Mugihiko/GlobalShape/Behav/DNNObjectRec_Kato2026';
addpath([rootD '/codes'])
savedir   = [rootD '/analysis/FigS6_RSA'];
if ~exist(savedir,'dir');mkdir(savedir);end

cnntypes = {'convnextL'  'clip_convnextL_image' 'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384'...
    'vit_l_16' 'clip_vit-l-laion_image'  'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k' ...
    'resnet50'  'resnet50-blur-st' 'resnet50-sin' ...
    'cornet-s' 'cornet-s-blur-st'};
imtype    = {'Ori' 'Fil' 'Lin' 'SilTex' 'Sil' 'Tex' 'LinFil'};
isGS      = [1 0 1 1 1 0 0]; % global shape
isTx      = [1 1 0 1 0 1 0]; % texture
isIP      = [1 1 1 0 0 0 1]; % internal parts

features = [isGS;isTx;isIP];
% model RDMs
figure
for f = 1:size(features,1)
    feat = features(f,:);
    rdm_tmp = nan(nchoosek(length(feat),2),1);
    k = 1;
    for i = 1:length(feat)-1
        for j = i+1:length(feat)
            rdm_tmp(k) =  feat(i) ~= 1 || feat(j) ~=1;
            k = k + 1;
        end
    end
    subplot(1,3,f)
    imagesc(squareform(rdm_tmp))
    axis square
    if f == 1
        modelRDM = rdm_tmp';
    else
        modelRDM = cat(1,modelRDM,rdm_tmp');
    end
end

for cnn = 1:length(cnntypes)
    cnntype = cnntypes{cnn};
    if ~exist([savedir '/' cnntype '.mat'],'file')
        layers = get_layer_names(cnntype,1);
        exemplars = dir([rootD '/rawdata/act/' cnntype '_Ori/*_o']);
        exemplars = cellfun(@(x) x(1:end-2),{exemplars.name},'UniformOutput',false)';

        r = nan(length(exemplars),size(features,1),length(layers));

        for lay = 1:length(layers)
            layer = layers{lay};
            for ex = 1:length(exemplars)
                for t = 1:length(imtype)
                    actD = sprintf([rootD '/rawdata/act/%s_%s'], cnntype, imtype{t});
                    act_file = dir([actD '/' exemplars{ex} '*/' layer '.mat']);
                    load([act_file.folder '/' act_file.name]);
                    if t == 1
                        actpat_all = nan(length(imtype),length(actpat), 'single');
                    end
                    actpat_all(t, :) = actpat;
                end
                rdm = pdist(actpat_all,'correlation');

                for f = 1:size(features,1)
                    r(ex,f,lay) = corr(rdm',modelRDM(f,:)','type','Spearman');
                end
            end
        end
        save([savedir '/' cnntype  '.mat'],'r','layers')
    end
end


% plot
cnntypes = {{'convnextL'  'clip_convnextL_image' 'convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384'}...
    {'vit_l_16' 'clip_vit-l-laion_image'  'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k' }...
    {'resnet50'  'resnet50-blur-st' 'resnet50-sin'} ...
    {'cornet-s' 'cornet-s-blur-st'}};
cnnlabels = {{'inConvNeXt'  'zsConvNeXt' 'ftConvNeXt'}...
    {'inViT' 'zsViT'  'ftViT' }...
    {'inResNet'  'blResNet' 'stResNet'} ...
    {'inCORnet' 'blCORnet'}};
titleL = {'convNext' 'ViT-L' 'resnet50' 'cornet-s'};
marks = {{'o' 'diamond' 'square'} {'o' 'diamond' 'square'} {'o' '^' 'v'} {'o' '^'}};
figlabel = {'Fig.S6b' 'Fig.S6c' 'Fig.S6d' 'Fig.S6e'};
cols = {[0 48 73;247 127 0; 214 40 40]/255 [0 48 73;247 127 0; 214 40 40]/255 ...
    [0 48 73;87 204 153; 199 125 255]/255 [0 48 73;87 204 153]/255};
layerlab = {'actF' 'actM' 'actL' 'penu'};
vars_tmp = {'Shape' 'Texture' 'Internal parts'};
startcell = {'A' 'F' 'K'};

jit = [.175 0 -.175];
figW = 15;
figH = 3.85;

P_two = cell(length(cnntypes),3);
T_two = cell(length(cnntypes),3);
for comb = 1:length(cnntypes)
    mark    = marks{comb};
    cnntype = cnntypes{comb};
    col     = cols{comb};
    cnnlabel = cnnlabels{comb};
    for cnn = 1:length(cnntype)
        load([savedir '/' cnntype{cnn}  '.mat'])
        if cnn == 1
            r_z_all = atanh(r);
        else
            r_z_all = cat(4,r_z_all,atanh(r));
        end
    end
    figure
    for cnn = 1:length(cnntype) % cnn = 2
        r_z = r_z_all(:,:,:,cnn);
        sd     = tanh(squeeze(std(r_z,0,1)));
        p_one = nan(size(r,2),size(r,3));
        t_one = nan(size(r,2),size(r,3));
        p_two = nan(size(r,2),size(r,3));
        t_two = nan(size(r,2),size(r,3));
        for mod = 1:size(r,2)
            subplot(1,3,mod);hold on
            if cnn == 1
                plot([0 size(r,3)],[0 0],'--','Color',[.5 .5 .5]);
                box off
                xticklabels(layerlab)
                xticks(1:length(layerlab))
            end
            r_avg = squeeze(tanh(mean(r_z(:,mod,:),1,'omitnan')))';

            xlim([0.5 size(r,3)+0.5])
            ylim([-0.55 1])
            for l = 1:length(r_avg)
                [~,p_one(mod,l),~,s] = ttest(r_z(:,mod,l),0,'tail','right');
                t_one(mod,l) = s.tstat;
                if cnn == 1
                    [~,p_two(mod,l),~,s] = ttest(r_z_all(:,mod,l,1),r_z_all(:,mod,l,2));
                elseif cnn == 2 & length(cnntype)>=3
                    [~,p_two(mod,l),~,s] = ttest(r_z_all(:,mod,l,2),r_z_all(:,mod,l,3));
                elseif cnn == 3
                    [~,p_two(mod,l),~,s] = ttest(r_z_all(:,mod,l,1),r_z_all(:,mod,l,3));
                end
                if ~isempty(s)
                    t_two(mod,l) = s.tstat;
                end
            end

            h = p_one(mod,:)*length(r_avg)<0.05;
            if any(h)
                text(find(h)-jit(cnn),repmat(.98,1,length(find(h))),'*','HorizontalAlignment','center','FontSize',10,'Color',col(cnn,:))
            end
            h_p = find(p_two(mod,:)*length(r_avg)<0.05);

            if any(p_two(mod,:))
                plot([-0.5 length(r_avg)+0.5],[-0.45+jit(cnn)/3 -0.45+jit(cnn)/3],'Color',[.5 .5 .5]+jit(cnn)*2,'LineWidth',0.5)
            end
            for si = 1:length(h_p)
                plot([h_p(si)-0.5 h_p(si)+0.5],[-0.45+jit(cnn)/3 -0.45+jit(cnn)/3],'Color',[.5 .5 .5]+jit(cnn)*2,'LineWidth',2.5)
            end

            plot((1:length(r_avg))-jit(cnn),r_avg,'Color',[col(cnn,:) 0.7],'LineWidth',2);
            scatter((1:length(r_avg))-jit(cnn),r_avg,30,mark{cnn},'MarkerEdgeColor','none',...
                'MarkerFaceColor',col(cnn,:),'MarkerFaceAlpha',0.8)

            errorbar((1:length(r_avg))-jit(cnn),r_avg,sd(mod,:),'Color',[col(cnn,:) 0.7],'LineStyle','none','LineWidth',.75)
            ax = gca;
            ax.TickDir = 'out';
            ax.LineWidth = 1;
            ax.FontName = 'Helvetica';
            ax.FontSize = 6;
            ax.XTickLabelRotation = 0;
            ylabel('Correlation')
        end

        if cnn == 1
            p_sd = p_one*length(r_avg); % p for source data
            t_sd = t_one;
            r_sd = squeeze(tanh(mean(r_z,1,'omitnan')));
            l_sd = cellfun(@(x) [cnnlabel{cnn} '-' x],layerlab,'UniformOutput',false)';
        else
            p_sd = cat(3,p_sd,p_one*length(r_avg));
            t_sd = cat(3,t_sd,t_one);
            r_sd = cat(3,r_sd,squeeze(tanh(mean(r_z,1,'omitnan'))));
            l_sd = cat(1,l_sd,cellfun(@(x) [cnnlabel{cnn} '-' x],layerlab,'UniformOutput',false)');
        end
        P_two{comb,cnn} = p_two*length(r_avg);
        T_two{comb,cnn} = t_two;
    end

    rectangle('Position',[1 1 figW figH],'EdgeColor','none','FaceColor','none');
    set(gcf,'Color','white','Units', 'centimeters', 'Position', [1 1 figW figH], ...
        'PaperUnits', 'centimeters','defaultAxesXColor','k','defaultAxesYColor','k',...
        'defaultAxesZColor','k','PaperPosition', [0 0 figW figH], 'PaperSize',[figW figH],'PaperPositionMode','auto')
    exportgraphics(gcf, [savedir '/' titleL{comb} '.pdf'], 'ContentType', 'vector');
    saveas(gcf,[savedir '/' figlabel{comb} '.png'])

    for i = 1:length(vars_tmp)
        t = cell(length(l_sd)+2,4);
        t(3:end,1) = l_sd;
        t{1,1} = vars_tmp{i};
        t{1,3} = 'one-sample test';
        t{2,2} = 'correlation';
        t{2,3} = 't(239)';
        t{2,4} = 'P (Bonferroni-corrected)';
        t(3:end,2) = num2cell(reshape(r_sd(i,:,:),[],1));
        t(3:end,3) = num2cell(reshape(t_sd(i,:,:),[],1));
        t(3:end,4) = num2cell(reshape(p_sd(i,:,:),[],1));
        writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{comb},...
            'WriteVariableNames',false,'Range',[startcell{i} num2str(1)]);


        t = cell(numel(p_two)+2,3);
        t{1,2} = 'paired test';
        t{2,2} = 't(239)';
        t{2,3} = 'P (Bonferroni-corrected)';
        m = 1;

        for m1 = 1:length(cnntype)-1
            for m2 = m1+1:length(cnntype)
                for l = 1:length(layerlab)
                    t{m+2,1} = [cnnlabel{m1} ' vs ' cnnlabel{m2} '-' layerlab{l}];
                    t{m+2,2} = T_two{comb,ceil(m/length(layerlab))}(i,l);
                    t{m+2,3} = P_two{comb,ceil(m/length(layerlab))}(i,l);
                    m = m +1;
                end
            end
        end
        if length(cnnlabel) == 3
            tmp = t(7:10,1);
            t(7:10,1) = t(11:14,1);
            t(11:14,1) = tmp;
        end
        writetable(cell2table(t), [rootD '/sourcedat.xlsx'],'Sheet',figlabel{comb},...
            'WriteVariableNames',false,'Range',[startcell{i} num2str(length(l_sd)+4)]);
    end
end