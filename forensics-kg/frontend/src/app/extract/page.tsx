'use client';

import { useState, useRef } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useExtractText, useExtractImage, useExtractCombined } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';
import type { ExtractionResult } from '@/types';
import { NODE_COLORS } from '@/types';

const DOC_TYPES = [
  { value: 'fir', label: 'FIR Report' },
  { value: 'court_judgment', label: 'Court Judgment' },
  { value: 'postmortem', label: 'Post-Mortem Report' },
  { value: 'lab_report', label: 'Forensic Lab Report' },
  { value: 'witness_deposition', label: 'Witness Deposition' },
  { value: 'soco_report', label: 'SOCO Report' },
];

const IMAGE_TYPES = [
  { value: 'bloodstain', label: 'Bloodstain Pattern' },
  { value: 'fingerprint', label: 'Fingerprint' },
  { value: 'wound', label: 'Wound Analysis' },
  { value: 'ballistics', label: 'Ballistics' },
  { value: 'document_forensics', label: 'Document Forensics' },
  { value: 'tool_marks', label: 'Tool Marks' },
];

export default function ExtractPage() {
  const [tab, setTab] = useState<'text' | 'image' | 'combined'>('text');
  const [text, setText] = useState('');
  const [sourceType, setSourceType] = useState('fir');
  const [imageType, setImageType] = useState('bloodstain');
  const [combinedText, setCombinedText] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageName, setImageName] = useState('');
  const [combinedFile, setCombinedFile] = useState<File | null>(null);
  const [combinedPreview, setCombinedPreview] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const combinedFileRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const { toast } = useToast();

  const selectedModel = useAppStore((s) => s.selectedModel);
  const textMutation = useExtractText();
  const imageMutation = useExtractImage();
  const combinedMutation = useExtractCombined();

  // Show results only for the active tab's mutation
  const result: ExtractionResult | undefined =
    tab === 'text' ? textMutation.data :
    tab === 'image' ? imageMutation.data :
    combinedMutation.data;
  const isLoading = textMutation.isPending || imageMutation.isPending || combinedMutation.isPending;

  const handleTextExtract = () => {
    if (!text.trim()) return;
    textMutation.mutate({ text, sourceType, store: true, model: selectedModel }, {
      onSuccess: () => toast('Extraction complete', 'success'),
      onError: (e: any) => toast(e.response?.data?.detail || 'Extraction failed', 'error'),
    });
  };

  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

  const processImage = (file: File) => {
    if (file.size > MAX_FILE_SIZE) {
      toast(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max: 50 MB.`, 'error');
      return;
    }
    setImageName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('image_type', imageType);
    formData.append('store_in_graph', 'true');
    imageMutation.mutate(formData, {
      onSuccess: () => toast('Image analysis complete', 'success'),
      onError: (e: any) => toast(e.response?.data?.detail || 'Image analysis failed', 'error'),
    });
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processImage(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      if (tab === 'combined') {
        setCombinedFile(file);
        const reader = new FileReader();
        reader.onload = (ev) => setCombinedPreview(ev.target?.result as string);
        reader.readAsDataURL(file);
      } else {
        processImage(file);
      }
    } else {
      toast('Please drop an image file', 'error');
    }
  };

  const handleCombinedImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > MAX_FILE_SIZE) {
        toast(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max: 50 MB.`, 'error');
        return;
      }
      setCombinedFile(file);
      const reader = new FileReader();
      reader.onload = (ev) => setCombinedPreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleCombinedExtract = () => {
    if (!combinedText.trim() || !combinedFile) return;
    const formData = new FormData();
    formData.append('file', combinedFile);
    formData.append('text', combinedText);
    formData.append('source_type', sourceType);
    formData.append('image_type', imageType);
    formData.append('store_in_graph', 'true');
    if (selectedModel) formData.append('model', selectedModel);
    combinedMutation.mutate(formData, {
      onSuccess: () => toast('Combined extraction complete', 'success'),
      onError: (e: any) => toast(e.response?.data?.detail || 'Combined extraction failed', 'error'),
    });
  };

  const tabLabels = { text: 'Document Text', image: 'Forensic Image', combined: 'Combined' };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Extract Entities & Relationships</h1>
      <p className="text-text-muted mb-6">Upload forensic documents or images for LLM-powered entity extraction</p>

      <div className="flex items-center justify-between mb-6">
        <div className="flex gap-2">
        {(['text', 'image', 'combined'] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              tab === t ? 'bg-[var(--primary)] text-white' : 'bg-surface-hover text-text-muted hover:bg-surface-active'
            }`}
          >
            {tabLabels[t]}
          </button>
        ))}
        </div>
        <ModelSelector />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <Card>
          {tab === 'text' ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-text-secondary">Document Text</h3>
                <select
                  value={sourceType}
                  onChange={(e) => setSourceType(e.target.value)}
                  className="bg-surface-input border border-border-default rounded-lg px-3 py-1.5 text-xs text-text-secondary focus:outline-none focus:border-[var(--primary)]/50"
                  aria-label="Document type"
                >
                  {DOC_TYPES.map((dt) => (
                    <option key={dt.value} value={dt.value}>{dt.label}</option>
                  ))}
                </select>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={`Paste ${DOC_TYPES.find(d => d.value === sourceType)?.label || 'document'} text here...`}
                aria-label="Document text"
                className="w-full h-64 bg-surface-input border border-border-default rounded-lg p-3 text-sm text-text-body resize-none focus:outline-none focus:border-[var(--primary)]/50"
              />
              <button
                type="button"
                onClick={handleTextExtract}
                disabled={isLoading || !text.trim()}
                className="mt-3 px-6 py-2.5 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-[var(--primary)]/80 transition flex items-center gap-2"
              >
                {textMutation.isPending && <Spinner size="sm" />}
                {textMutation.isPending ? 'Extracting...' : 'Extract Entities'}
              </button>
            </>
          ) : tab === 'image' ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-text-secondary">Upload Forensic Image</h3>
                <select
                  value={imageType}
                  onChange={(e) => setImageType(e.target.value)}
                  className="bg-surface-input border border-border-default rounded-lg px-3 py-1.5 text-xs text-text-secondary focus:outline-none focus:border-[var(--primary)]/50"
                  aria-label="Image type"
                >
                  {IMAGE_TYPES.map((it) => (
                    <option key={it.value} value={it.value}>{it.label}</option>
                  ))}
                </select>
              </div>
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileRef.current?.click()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition ${
                  dragOver ? 'border-[var(--primary)] bg-[var(--primary)]/5' : 'border-border-default hover:border-border-hover'
                }`}
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                  aria-label="Upload forensic image"
                />
                {imagePreview ? (
                  <div>
                    <Image src={imagePreview} alt="Preview" width={320} height={160} className="max-h-40 mx-auto rounded mb-2 w-auto" unoptimized />
                    <p className="text-xs text-text-muted">{imageName}</p>
                  </div>
                ) : (
                  <>
                    <p className="text-text-muted mb-2">Click or drag & drop image here</p>
                    <p className="text-xs text-text-ghost">JPG, PNG, TIFF, BMP up to 50MB</p>
                  </>
                )}
              </div>
              {imageMutation.isPending && (
                <div className="mt-3 flex items-center gap-2 text-sm text-[var(--primary)]">
                  <Spinner size="sm" /> Analyzing image...
                </div>
              )}
            </>
          ) : (
            <>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-text-secondary">Combined Text + Image Analysis</h3>
                <div className="flex gap-2">
                  <select
                    value={sourceType}
                    onChange={(e) => setSourceType(e.target.value)}
                    className="bg-surface-input border border-border-default rounded-lg px-2 py-1.5 text-xs text-text-secondary focus:outline-none focus:border-[var(--primary)]/50"
                    aria-label="Document type for combined"
                  >
                    {DOC_TYPES.map((dt) => (
                      <option key={dt.value} value={dt.value}>{dt.label}</option>
                    ))}
                  </select>
                  <select
                    value={imageType}
                    onChange={(e) => setImageType(e.target.value)}
                    className="bg-surface-input border border-border-default rounded-lg px-2 py-1.5 text-xs text-text-secondary focus:outline-none focus:border-[var(--primary)]/50"
                    aria-label="Image type for combined"
                  >
                    {IMAGE_TYPES.map((it) => (
                      <option key={it.value} value={it.value}>{it.label}</option>
                    ))}
                  </select>
                </div>
              </div>
              <textarea
                value={combinedText}
                onChange={(e) => setCombinedText(e.target.value)}
                placeholder={`Paste ${DOC_TYPES.find(d => d.value === sourceType)?.label || 'document'} text here...`}
                aria-label="Document text for combined analysis"
                className="w-full h-32 bg-surface-input border border-border-default rounded-lg p-3 text-sm text-text-body resize-none focus:outline-none focus:border-[var(--primary)]/50 mb-3"
              />
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => combinedFileRef.current?.click()}
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
                  dragOver ? 'border-[var(--primary)] bg-[var(--primary)]/5' : 'border-border-default hover:border-border-hover'
                }`}
              >
                <input
                  ref={combinedFileRef}
                  type="file"
                  accept="image/*"
                  onChange={handleCombinedImageUpload}
                  className="hidden"
                  aria-label="Upload forensic image for combined analysis"
                />
                {combinedPreview ? (
                  <div>
                    <Image src={combinedPreview} alt="Preview" width={320} height={112} className="max-h-28 mx-auto rounded mb-2 w-auto" unoptimized />
                    <p className="text-xs text-text-muted">{combinedFile?.name}</p>
                  </div>
                ) : (
                  <>
                    <p className="text-text-muted mb-1 text-sm">Drop forensic image here</p>
                    <p className="text-xs text-text-ghost">JPG, PNG, TIFF, BMP up to 50MB</p>
                  </>
                )}
              </div>
              <button
                type="button"
                onClick={handleCombinedExtract}
                disabled={isLoading || !combinedText.trim() || !combinedFile}
                className="mt-3 px-6 py-2.5 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-[var(--primary)]/80 transition flex items-center gap-2"
              >
                {combinedMutation.isPending && <Spinner size="sm" />}
                {combinedMutation.isPending ? 'Analyzing...' : 'Run Combined Analysis'}
              </button>
            </>
          )}
        </Card>

        {/* Results Panel */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-text-secondary">Extraction Results</h3>
            {result && (
              <button
                type="button"
                onClick={() => router.push('/graph')}
                className="text-xs text-[var(--primary)] hover:underline"
              >
                View in Graph &rarr;
              </button>
            )}
          </div>

          {(textMutation.error || imageMutation.error || combinedMutation.error) && (
            <div className="bg-red-500/10 border border-red-500/30 rounded p-3 text-sm text-red-400 mb-3">
              {(textMutation.error as any)?.response?.data?.detail || (imageMutation.error as any)?.response?.data?.detail || (combinedMutation.error as any)?.response?.data?.detail || 'Extraction failed'}
            </div>
          )}

          {result ? (
            <div className="space-y-4 max-h-[500px] overflow-y-auto">
              <div className="flex gap-4 text-xs text-text-faint">
                <span>{result.entities.length} entities</span>
                <span>{result.relationships.length} relationships</span>
                <span>{result.processing_time_ms.toFixed(0)}ms</span>
              </div>

              {/* Synthesis metadata for combined results */}
              {result.metadata?.unified_assessment && (
                <div className="bg-[var(--primary)]/10 border border-[var(--primary)]/30 rounded-lg p-3">
                  <h4 className="text-xs uppercase tracking-wider text-[var(--primary)] mb-1">Multi-modal Synthesis</h4>
                  <p className="text-xs text-text-secondary">{result.metadata.unified_assessment}</p>
                  <div className="flex gap-3 mt-2 text-xs text-text-faint">
                    <span>Text: {result.metadata.text_entities} entities</span>
                    <span>Image: {result.metadata.image_entities} entities</span>
                    <span>Cross-refs: {result.metadata.cross_references}</span>
                  </div>
                </div>
              )}

              <div>
                <h4 className="text-xs uppercase tracking-wider text-text-faint mb-2">Entities</h4>
                <div className="space-y-2">
                  {result.entities.map((e, i) => (
                    <div key={i} className="bg-surface-inset rounded p-2.5 border-l-2" style={{ borderColor: NODE_COLORS[e.entity_type] || '#6b7280' }}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs px-1.5 py-0.5 rounded text-text-primary" style={{ backgroundColor: (NODE_COLORS[e.entity_type] || '#6b7280') + '30' }}>
                          {e.entity_type}
                        </span>
                        <span className="text-xs text-text-faint">{e.entity_id}</span>
                      </div>
                      <div className="text-xs text-text-tertiary">
                        {Object.entries(e.properties || {}).slice(0, 4).map(([k, v]) => (
                          <span key={k} className="mr-3">{k}: {String(v).slice(0, 50)}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-xs uppercase tracking-wider text-text-faint mb-2">Relationships</h4>
                <div className="space-y-1">
                  {result.relationships.map((r, i) => (
                    <div key={i} className="text-xs bg-surface-inset rounded p-2">
                      <span className="text-[var(--primary)]">{r.source_entity_id}</span>
                      <span className="text-text-ghost mx-2">--[{r.relationship_type}]--&gt;</span>
                      <span className="text-[var(--primary)]">{r.target_entity_id}</span>
                      <span className="text-text-whisper ml-2">({(r.confidence * 100).toFixed(0)}%)</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-text-ghost">Results will appear here after extraction.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
