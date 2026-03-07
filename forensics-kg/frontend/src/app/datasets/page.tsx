'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDatasets, useGenerateFIR, useProcessBloodstain, useIngestJudgments, useGenerateGoldBatch, useGenerateDocuments, useDownloadForensicData, useGenerateImages, useProcessImages, useImageCounts, useProcessJudgments } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner, PageSpinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { DataTable } from '@/components/ui/data-table';
import { EmptyState } from '@/components/ui/empty-state';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';

const DOC_GEN_TYPES = [
  { value: 'postmortem', label: 'Post-Mortem Reports' },
  { value: 'lab_report', label: 'Lab Reports' },
  { value: 'witness_deposition', label: 'Witness Depositions' },
  { value: 'soco_report', label: 'SOCO Reports' },
];

export default function DatasetsPage() {
  const router = useRouter();
  const { toast } = useToast();
  const selectedModel = useAppStore((s) => s.selectedModel);
  const [firCount, setFirCount] = useState(10);
  const [docGenType, setDocGenType] = useState('postmortem');
  const [docGenCount, setDocGenCount] = useState(10);
  const [judgmentLimit, setJudgmentLimit] = useState(20);
  const [judgmentSource, setJudgmentSource] = useState('huggingface');
  const [confirmGenerate, setConfirmGenerate] = useState(false);
  const [confirmBloodstain, setConfirmBloodstain] = useState(false);
  const [confirmJudgments, setConfirmJudgments] = useState(false);
  const [confirmDocGen, setConfirmDocGen] = useState(false);
  const [confirmGold, setConfirmGold] = useState(false);
  const [confirmDownload, setConfirmDownload] = useState(false);
  const [confirmImageGen, setConfirmImageGen] = useState(false);
  const [imageGenType, setImageGenType] = useState('ballistics');
  const [imageGenCount, setImageGenCount] = useState(20);
  const [confirmProcessImages, setConfirmProcessImages] = useState(false);
  const [processImageType, setProcessImageType] = useState('fingerprint');
  const [processImageLimit, setProcessImageLimit] = useState(10);
  const [confirmProcessJudgments, setConfirmProcessJudgments] = useState(false);
  const [processJudgmentLimit, setProcessJudgmentLimit] = useState(20);

  const { data: datasets, isLoading, isFetching } = useDatasets();
  const generateMutation = useGenerateFIR();
  const bloodstainMutation = useProcessBloodstain();
  const ingestMutation = useIngestJudgments();
  const goldBatchMutation = useGenerateGoldBatch();
  const docGenMutation = useGenerateDocuments();
  const downloadMutation = useDownloadForensicData();
  const imageGenMutation = useGenerateImages();
  const processImagesMutation = useProcessImages();
  const { data: imageCounts } = useImageCounts();
  const processJudgmentsMutation = useProcessJudgments();

  const handleGenerateFIR = () => {
    setConfirmGenerate(false);
    generateMutation.mutate({ count: firCount }, {
      onSuccess: (res: any) => {
        toast(`Generated ${res.generated} FIRs, processed ${res.processed?.length || 0} into graph`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Generation failed', 'error'),
    });
  };

  const handleProcessBloodstain = () => {
    setConfirmBloodstain(false);
    bloodstainMutation.mutate(5, {
      onSuccess: (res: any) => {
        toast(`Processed ${res.processed?.length || 0} experiments (${res.errors?.length || 0} errors)`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Processing failed', 'error'),
    });
  };

  const handleIngestJudgments = () => {
    setConfirmJudgments(false);
    ingestMutation.mutate({ source: judgmentSource, limit: judgmentLimit }, {
      onSuccess: (res: any) => {
        toast(`Ingested ${res.count || 0} judgments from ${judgmentSource}`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Ingestion failed', 'error'),
    });
  };

  const handleGenerateGoldBatch = () => {
    setConfirmGold(false);
    goldBatchMutation.mutate({ limit: 20, model: selectedModel }, {
      onSuccess: (res: any) => {
        toast(`Generated ${res.count || 0} gold standards`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Gold generation failed', 'error'),
    });
  };

  const handleGenerateDocuments = () => {
    setConfirmDocGen(false);
    docGenMutation.mutate({ docType: docGenType, count: docGenCount }, {
      onSuccess: (res: any) => {
        toast(`Generated ${res.count || 0} ${DOC_GEN_TYPES.find(d => d.value === docGenType)?.label || 'documents'}`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Generation failed', 'error'),
    });
  };

  const handleDownloadForensicData = () => {
    setConfirmDownload(false);
    downloadMutation.mutate({ datasetType: 'all', limit: 20 }, {
      onSuccess: (res: any) => {
        toast(`Downloaded forensic data: ${res.total_items || 0} items across datasets`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Download failed', 'error'),
    });
  };

  const handleGenerateImages = () => {
    setConfirmImageGen(false);
    imageGenMutation.mutate({ imageType: imageGenType, count: imageGenCount }, {
      onSuccess: (res: any) => {
        toast(`Generated ${res.count || 0} ${imageGenType} images`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Image generation failed', 'error'),
    });
  };

  const handleProcessImages = () => {
    setConfirmProcessImages(false);
    processImagesMutation.mutate({ imageType: processImageType, limit: processImageLimit }, {
      onSuccess: (res: any) => {
        toast(`Processed ${res.processed?.length || 0} ${processImageType} images (${res.errors?.length || 0} errors)`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Processing failed', 'error'),
    });
  };

  const handleProcessJudgments = () => {
    setConfirmProcessJudgments(false);
    processJudgmentsMutation.mutate(processJudgmentLimit, {
      onSuccess: (res: any) => {
        toast(`Extracted ${res.processed?.length || 0} judgments into graph (${res.errors?.length || 0} errors)`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Processing failed', 'error'),
    });
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Datasets</h1>
      <p className="text-text-muted mb-6">Manage forensic datasets and generate synthetic data</p>

      {/* Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Generate Synthetic FIR Reports</h3>
          <p className="text-xs text-text-faint mb-4">
            Use GPT-4o to generate realistic First Information Reports with ground truth data.
          </p>
          <div className="flex items-center gap-3">
            <label htmlFor="fir-count" className="text-xs text-text-muted">Count:</label>
            <input
              id="fir-count"
              type="number"
              min={1}
              max={100}
              value={firCount}
              onChange={(e) => setFirCount(Number(e.target.value))}
              className="w-20 bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
            />
            <button
              type="button"
              onClick={() => setConfirmGenerate(true)}
              disabled={generateMutation.isPending}
              className="px-4 py-2 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
            >
              {generateMutation.isPending && <Spinner size="sm" />}
              {generateMutation.isPending ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Ingest Court Judgments</h3>
          <p className="text-xs text-text-faint mb-4">
            Ingest real Indian criminal case judgments from HuggingFace, CSV, or Indian Kanoon API.
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            <select
              value={judgmentSource}
              onChange={(e) => setJudgmentSource(e.target.value)}
              className="bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
              aria-label="Judgment source"
            >
              <option value="huggingface">HuggingFace (auto-download)</option>
              <option value="csv">CSV (Kaggle)</option>
              <option value="indian_kanoon">Indian Kanoon API</option>
            </select>
            <label htmlFor="judgment-limit" className="text-xs text-text-muted">Limit:</label>
            <input
              id="judgment-limit"
              type="number"
              min={1}
              max={100}
              value={judgmentLimit}
              onChange={(e) => setJudgmentLimit(Number(e.target.value))}
              className="w-16 bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
            />
            <button
              type="button"
              onClick={() => setConfirmJudgments(true)}
              disabled={ingestMutation.isPending}
              className="px-4 py-2 bg-[#2563eb] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
            >
              {ingestMutation.isPending && <Spinner size="sm" />}
              {ingestMutation.isPending ? 'Ingesting...' : 'Ingest'}
            </button>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Generate Gold Standards</h3>
          <p className="text-xs text-text-faint mb-4">
            Generate gold standard annotations from ingested court judgments using LLM.
          </p>
          <div className="flex items-center gap-3">
            <ModelSelector />
            <button
              type="button"
              onClick={() => setConfirmGold(true)}
              disabled={goldBatchMutation.isPending}
              className="px-4 py-2 bg-[#f59e0b] text-black rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
            >
              {goldBatchMutation.isPending && <Spinner size="sm" />}
              {goldBatchMutation.isPending ? 'Generating...' : 'Generate Batch'}
            </button>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Generate Forensic Documents</h3>
          <p className="text-xs text-text-faint mb-4">
            Generate synthetic forensic documents of various types with ground truth.
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            <select
              value={docGenType}
              onChange={(e) => setDocGenType(e.target.value)}
              className="bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
              aria-label="Document type to generate"
            >
              {DOC_GEN_TYPES.map((dt) => (
                <option key={dt.value} value={dt.value}>{dt.label}</option>
              ))}
            </select>
            <label htmlFor="doc-count" className="text-xs text-text-muted">Count:</label>
            <input
              id="doc-count"
              type="number"
              min={1}
              max={50}
              value={docGenCount}
              onChange={(e) => setDocGenCount(Number(e.target.value))}
              className="w-16 bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
            />
            <button
              type="button"
              onClick={() => setConfirmDocGen(true)}
              disabled={docGenMutation.isPending}
              className="px-4 py-2 bg-[#14b8a6] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
            >
              {docGenMutation.isPending && <Spinner size="sm" />}
              {docGenMutation.isPending ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Process Bloodstain Data</h3>
          <p className="text-xs text-text-faint mb-4">
            Process existing bloodstain experiment images and metadata through the extraction pipeline.
          </p>
          <button
            type="button"
            onClick={() => setConfirmBloodstain(true)}
            disabled={bloodstainMutation.isPending}
            className="px-4 py-2 bg-[#be123c] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {bloodstainMutation.isPending && <Spinner size="sm" />}
            {bloodstainMutation.isPending ? 'Processing...' : 'Process (5 experiments)'}
          </button>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Download Real Forensic Datasets</h3>
          <p className="text-xs text-text-faint mb-4">
            Auto-download real forensic datasets: SOCOFing fingerprints (Kaggle), CEDAR signatures, AZH wound images, Mendeley autopsy reports, Multi-LexSum depositions.
          </p>
          <button
            type="button"
            onClick={() => setConfirmDownload(true)}
            disabled={downloadMutation.isPending}
            className="px-4 py-2 bg-[#7c3aed] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {downloadMutation.isPending && <Spinner size="sm" />}
            {downloadMutation.isPending ? 'Downloading...' : 'Download All'}
          </button>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Generate Synthetic Forensic Images</h3>
          <p className="text-xs text-text-faint mb-4">
            Generate procedural forensic images (ballistics striations, cartridge headstamps, tool marks) with ground truth metadata using OpenCV.
          </p>
          <div className="flex items-center gap-3 mb-3">
            <select
              value={imageGenType}
              onChange={(e) => setImageGenType(e.target.value)}
              aria-label="Forensic image type"
              className="bg-surface-hover border border-border-default rounded-lg px-3 py-1.5 text-sm text-text-body"
            >
              <option value="ballistics">Ballistics</option>
              <option value="tool_marks">Tool Marks</option>
            </select>
            <input
              type="number"
              min={1}
              max={100}
              value={imageGenCount}
              onChange={(e) => setImageGenCount(Number(e.target.value))}
              aria-label="Number of images to generate"
              className="bg-surface-hover border border-border-default rounded-lg px-3 py-1.5 text-sm text-text-body w-20"
            />
          </div>
          <button
            type="button"
            onClick={() => setConfirmImageGen(true)}
            disabled={imageGenMutation.isPending}
            className="px-4 py-2 bg-[#0891b2] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {imageGenMutation.isPending && <Spinner size="sm" />}
            {imageGenMutation.isPending ? 'Generating...' : 'Generate Images'}
          </button>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Process Forensic Images Into Graph</h3>
          <p className="text-xs text-text-faint mb-4">
            Run downloaded/generated forensic images through CV + GPT-4o vision analysis and store entities in the knowledge graph.
          </p>
          <div className="flex items-center gap-3 mb-3 flex-wrap">
            <select
              value={processImageType}
              onChange={(e) => setProcessImageType(e.target.value)}
              aria-label="Forensic image type to process"
              className="bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
            >
              <option value="fingerprint">Fingerprints {imageCounts?.fingerprint ? `(${imageCounts.fingerprint})` : ''}</option>
              <option value="wound">Wounds {imageCounts?.wound ? `(${imageCounts.wound})` : ''}</option>
              <option value="document_forensics">Signatures {imageCounts?.document_forensics ? `(${imageCounts.document_forensics})` : ''}</option>
              <option value="ballistics">Ballistics {imageCounts?.ballistics ? `(${imageCounts.ballistics})` : ''}</option>
              <option value="tool_marks">Tool Marks {imageCounts?.tool_marks ? `(${imageCounts.tool_marks})` : ''}</option>
            </select>
            <label className="text-xs text-text-muted">Limit:</label>
            <input
              type="number"
              min={1}
              max={200}
              value={processImageLimit}
              onChange={(e) => setProcessImageLimit(Number(e.target.value))}
              aria-label="Number of images to process"
              className="w-20 bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
            />
          </div>
          <button
            type="button"
            onClick={() => setConfirmProcessImages(true)}
            disabled={processImagesMutation.isPending}
            className="px-4 py-2 bg-[#059669] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {processImagesMutation.isPending && <Spinner size="sm" />}
            {processImagesMutation.isPending ? 'Processing...' : 'Process Into Graph'}
          </button>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-3">Process Court Judgments Into Graph</h3>
          <p className="text-xs text-text-faint mb-4">
            Extract entities from ingested court judgment texts into the knowledge graph using NER pipeline.
          </p>
          <div className="flex items-center gap-3 mb-3">
            <label className="text-xs text-text-muted">Limit:</label>
            <input
              type="number"
              min={1}
              max={100}
              value={processJudgmentLimit}
              onChange={(e) => setProcessJudgmentLimit(Number(e.target.value))}
              aria-label="Number of judgments to process"
              className="w-20 bg-surface-input border border-border-default rounded px-2 py-1.5 text-sm text-text-body focus:outline-none"
            />
          </div>
          <button
            type="button"
            onClick={() => setConfirmProcessJudgments(true)}
            disabled={processJudgmentsMutation.isPending}
            className="px-4 py-2 bg-[#059669] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {processJudgmentsMutation.isPending && <Spinner size="sm" />}
            {processJudgmentsMutation.isPending ? 'Extracting...' : 'Extract Into Graph'}
          </button>
        </Card>
      </div>

      {/* Dataset List */}
      <Card padding={false}>
        <div className="p-4 border-b border-border-default flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-text-secondary">Available Datasets</h3>
            {isFetching && !isLoading && <Spinner size="sm" />}
          </div>
          {datasets && datasets.length > 0 && (
            <button
              type="button"
              onClick={() => router.push('/graph')}
              className="text-xs text-[var(--primary)] hover:underline"
            >
              View in Graph &rarr;
            </button>
          )}
        </div>
        <div className="p-4">
          {isLoading ? (
            <PageSpinner message="Loading datasets..." />
          ) : !datasets || datasets.length === 0 ? (
            <EmptyState
              icon="▦"
              title="No datasets"
              message="Generate synthetic FIR reports or process bloodstain data to create datasets."
            />
          ) : (
            <DataTable
              data={datasets}
              columns={[
                { key: 'name', header: 'Name' },
                { key: 'type', header: 'Type' },
                { key: 'record_count', header: 'Records', render: (d) => (
                  <span className="font-mono">{d.record_count}</span>
                )},
                { key: 'loaded', header: 'Status', render: (d) => (
                  <Badge variant={d.loaded ? 'success' : 'default'}>
                    {d.loaded ? 'Loaded' : 'Available'}
                  </Badge>
                )},
              ]}
              searchable
              searchKeys={['name', 'type']}
              emptyMessage="No datasets found"
            />
          )}
        </div>
      </Card>

      {/* Confirm Dialogs */}
      <ConfirmDialog
        open={confirmGenerate}
        title="Generate Synthetic FIRs"
        message={`This will generate ${firCount} synthetic FIR reports using GPT-4o and process them into the knowledge graph. This may take a few minutes.`}
        confirmLabel="Generate"
        onConfirm={handleGenerateFIR}
        onCancel={() => setConfirmGenerate(false)}
      />
      <ConfirmDialog
        open={confirmBloodstain}
        title="Process Bloodstain Data"
        message="This will process 5 bloodstain experiments through the image analysis pipeline. This may take a few minutes."
        confirmLabel="Process"
        onConfirm={handleProcessBloodstain}
        onCancel={() => setConfirmBloodstain(false)}
      />
      <ConfirmDialog
        open={confirmJudgments}
        title="Ingest Court Judgments"
        message={`This will ingest up to ${judgmentLimit} criminal case judgments from ${judgmentSource === 'huggingface' ? 'HuggingFace (auto-download, no auth needed)' : judgmentSource === 'csv' ? 'Kaggle CSV' : 'Indian Kanoon API'}. This may take several minutes.`}
        confirmLabel="Ingest"
        onConfirm={handleIngestJudgments}
        onCancel={() => setConfirmJudgments(false)}
      />
      <ConfirmDialog
        open={confirmGold}
        title="Generate Gold Standards"
        message="This will generate gold standard annotations for up to 20 ingested court judgments using the selected LLM model. This may take several minutes."
        confirmLabel="Generate"
        onConfirm={handleGenerateGoldBatch}
        onCancel={() => setConfirmGold(false)}
      />
      <ConfirmDialog
        open={confirmDocGen}
        title={`Generate ${DOC_GEN_TYPES.find(d => d.value === docGenType)?.label || 'Documents'}`}
        message={`This will generate ${docGenCount} synthetic ${DOC_GEN_TYPES.find(d => d.value === docGenType)?.label?.toLowerCase() || 'documents'} using GPT-4o with ground truth data. This may take a few minutes.`}
        confirmLabel="Generate"
        onConfirm={handleGenerateDocuments}
        onCancel={() => setConfirmDocGen(false)}
      />
      <ConfirmDialog
        open={confirmDownload}
        title="Download Real Forensic Datasets"
        message="This will download real forensic datasets: SOCOFing fingerprints (~6,000 images from Kaggle), CEDAR signatures (~2,640 images), AZH wound images (~400 images), Mendeley autopsy reports, and Multi-LexSum depositions. Downloads may take several minutes depending on your connection."
        confirmLabel="Download"
        onConfirm={handleDownloadForensicData}
        onCancel={() => setConfirmDownload(false)}
      />
      <ConfirmDialog
        open={confirmImageGen}
        title={`Generate ${imageGenType === 'ballistics' ? 'Ballistics' : 'Tool Mark'} Images`}
        message={`This will generate ${imageGenCount} synthetic ${imageGenType === 'ballistics' ? 'ballistics (striations, headstamps, comparisons)' : 'tool mark (striation, impression, cut, pry)'} images with ground truth metadata using OpenCV.`}
        confirmLabel="Generate"
        onConfirm={handleGenerateImages}
        onCancel={() => setConfirmImageGen(false)}
      />
      <ConfirmDialog
        open={confirmProcessImages}
        title={`Process ${processImageType.replace('_', ' ')} Images`}
        message={`This will process up to ${processImageLimit} ${processImageType.replace('_', ' ')} images through CV + GPT-4o vision analysis and store extracted entities in the knowledge graph. This uses the OpenAI API and may take several minutes.`}
        confirmLabel="Process"
        onConfirm={handleProcessImages}
        onCancel={() => setConfirmProcessImages(false)}
      />
      <ConfirmDialog
        open={confirmProcessJudgments}
        title="Process Court Judgments Into Graph"
        message={`This will extract entities from up to ${processJudgmentLimit} ingested court judgment texts using the NER pipeline and store them in the knowledge graph. This uses the OpenAI API and may take several minutes.`}
        confirmLabel="Extract"
        onConfirm={handleProcessJudgments}
        onCancel={() => setConfirmProcessJudgments(false)}
      />
    </div>
  );
}
