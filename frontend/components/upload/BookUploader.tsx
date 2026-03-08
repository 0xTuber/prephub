'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

interface BookUploaderProps {
  onUploadComplete: (filePath: string, fileName: string) => void
  compact?: boolean
}

export default function BookUploader({ onUploadComplete, compact = false }: BookUploaderProps) {
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    if (!file.name.endsWith('.pdf')) {
      setError('Please upload a PDF file')
      return
    }

    setUploading(true)
    setError(null)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      setUploadedFile(file.name)
      setUploadProgress(100)
      onUploadComplete(data.filePath, file.name)
    } catch (err) {
      setError('Failed to upload file. Please try again.')
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
    }
  }, [onUploadComplete])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    disabled: uploading
  })

  return (
    <div className={cn("w-full", !compact && "max-w-xl mx-auto")}>
      <div
        {...getRootProps()}
        className={cn(
          "bg-white dark:bg-gray-800 border-2 border-dashed rounded-2xl text-center cursor-pointer transition-all duration-300 shadow-sm",
          compact ? "p-6" : "p-10",
          isDragActive && "border-blue-400 bg-blue-50 dark:bg-blue-900/20 shadow-lg shadow-blue-500/10",
          uploading && "cursor-not-allowed opacity-60",
          !isDragActive && !uploading && "border-gray-200 hover:border-blue-300 hover:shadow-md dark:border-gray-700 dark:hover:border-blue-500"
        )}
      >
        <input {...getInputProps()} />

        {uploading ? (
          <div className={cn("space-y-4", compact && "space-y-3")}>
            <div className={cn(
              "mx-auto rounded-full bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center",
              compact ? "w-12 h-12" : "w-16 h-16"
            )}>
              <Loader2 className={cn("text-blue-600 animate-spin", compact ? "h-6 w-6" : "h-8 w-8")} />
            </div>
            <p className={cn("font-medium text-gray-900 dark:text-white", compact ? "text-base" : "text-lg")}>Uploading...</p>
            <Progress value={uploadProgress} className="h-2" indicatorClassName="bg-gradient-to-r from-blue-600 to-blue-400" />
          </div>
        ) : uploadedFile ? (
          <div className={cn("space-y-4", compact && "space-y-3")}>
            <div className={cn(
              "mx-auto rounded-full bg-green-50 dark:bg-green-900/30 flex items-center justify-center",
              compact ? "w-12 h-12" : "w-16 h-16"
            )}>
              <CheckCircle className={cn("text-green-500", compact ? "h-6 w-6" : "h-8 w-8")} />
            </div>
            <p className={cn("font-medium text-green-600 dark:text-green-400", compact ? "text-base" : "text-lg")}>Uploaded Successfully</p>
            <div className="flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400">
              <FileText className="h-4 w-4" />
              <span className="text-sm truncate max-w-[200px]">{uploadedFile}</span>
            </div>
            <Button
              variant="outline"
              size={compact ? "sm" : "default"}
              className="border-gray-200 dark:border-gray-700"
              onClick={(e) => {
                e.stopPropagation()
                setUploadedFile(null)
              }}
            >
              Upload Different File
            </Button>
          </div>
        ) : (
          <div className={cn("space-y-4", compact && "space-y-3")}>
            <div className={cn(
              "mx-auto rounded-full bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center",
              compact ? "w-12 h-12" : "w-16 h-16"
            )}>
              <Upload className={cn("text-blue-500", compact ? "h-6 w-6" : "h-8 w-8")} />
            </div>
            <div>
              <p className={cn("font-semibold text-gray-900 dark:text-white", compact ? "text-base" : "text-lg")}>
                {isDragActive ? "Drop your PDF here" : "Drag & drop a PDF book"}
              </p>
              <p className={cn("text-gray-500 dark:text-gray-400", compact ? "text-xs mt-1" : "text-sm mt-2")}>
                or click to browse files
              </p>
            </div>
            {!compact && (
              <div className="pt-2">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-xs text-gray-600 dark:text-gray-300">
                  <FileText className="h-3 w-3" />
                  PDF files only
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-center gap-3 text-red-600 dark:text-red-400">
          <AlertCircle className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}
    </div>
  )
}
