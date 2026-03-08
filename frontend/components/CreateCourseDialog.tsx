'use client'

import { useState } from 'react'
import { Sparkles } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import BookUploader from '@/components/upload/BookUploader'

interface CreateCourseDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onGenerate: (bookPath: string, courseName: string) => void
}

export default function CreateCourseDialog({
  open,
  onOpenChange,
  onGenerate,
}: CreateCourseDialogProps) {
  const [uploadedFile, setUploadedFile] = useState<{ path: string; name: string } | null>(null)
  const [courseName, setCourseName] = useState('')

  const handleUploadComplete = (filePath: string, fileName: string) => {
    setUploadedFile({ path: filePath, name: fileName })
    const baseName = fileName.replace('.pdf', '').replace(/[_-]/g, ' ')
    setCourseName(baseName)
  }

  const handleGenerate = () => {
    if (!uploadedFile || !courseName.trim()) return
    onGenerate(uploadedFile.path, courseName.trim())
    // Reset state for next use
    setUploadedFile(null)
    setCourseName('')
    onOpenChange(false)
  }

  const handleCancel = () => {
    setUploadedFile(null)
    setCourseName('')
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Create New Course</DialogTitle>
          <DialogDescription>
            Upload a PDF book and generate an interactive learning course.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Step 1: Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Step 1: Upload Book
            </label>
            <BookUploader onUploadComplete={handleUploadComplete} compact />
          </div>

          {/* Step 2: Course Name */}
          {uploadedFile && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Step 2: Course Name
              </label>
              <input
                type="text"
                value={courseName}
                onChange={(e) => setCourseName(e.target.value)}
                placeholder="Enter course name..."
                className="w-full px-4 py-3 border border-gray-200 dark:border-gray-700 rounded-xl bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all text-gray-900 dark:text-white"
              />
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button
            onClick={handleGenerate}
            disabled={!uploadedFile || !courseName.trim()}
            className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white"
          >
            <Sparkles className="h-4 w-4 mr-2" />
            Generate
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
