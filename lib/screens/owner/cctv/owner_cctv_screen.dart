import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../widgets/owner_feature_frame.dart';
import '../../../widgets/responsive_shell.dart';
import 'camera_view_screen.dart';

class OwnerCctvScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerCctvScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerCctvScreen> createState() => _OwnerCctvScreenState();
}

class _OwnerCctvScreenState extends State<OwnerCctvScreen> {
  String _selectedFilter = 'All';

  final List<String> _filters = [
    'All',
    'Online',
    'Offline',
    'Ground Floor',
    '1st Floor',
    '2nd Floor',
  ];

  final List<Map<String, dynamic>> _cameras = [
    {
      'name': 'Main Gate',
      'location': 'Ground Floor',
      'status': 'Online',
      'recording': true,
      'quality': 'HD',
      'lastUpdated': 'Live',
      'icon': Icons.door_front_door_rounded,
    },
    {
      'name': 'Reception',
      'location': 'Ground Floor',
      'status': 'Online',
      'recording': true,
      'quality': 'HD',
      'lastUpdated': 'Live',
      'icon': Icons.meeting_room_rounded,
    },
    {
      'name': 'Lobby Camera',
      'location': 'Ground Floor',
      'status': 'Offline',
      'recording': false,
      'quality': 'HD',
      'lastUpdated': '12 min ago',
      'icon': Icons.chair_alt_rounded,
    },
    {
      'name': 'Floor 1 Hallway',
      'location': '1st Floor',
      'status': 'Online',
      'recording': true,
      'quality': 'HD',
      'lastUpdated': 'Live',
      'icon': Icons.sensor_door_rounded,
    },
    {
      'name': 'Floor 1 Stairs',
      'location': '1st Floor',
      'status': 'Online',
      'recording': true,
      'quality': 'HD',
      'lastUpdated': 'Live',
      'icon': Icons.stairs_rounded,
    },
    {
      'name': 'Floor 2 Hallway',
      'location': '2nd Floor',
      'status': 'Online',
      'recording': true,
      'quality': 'HD',
      'lastUpdated': 'Live',
      'icon': Icons.sensor_door_rounded,
    },
  ];

  List<Map<String, dynamic>> get _filteredCameras {
    return _cameras.where((camera) {
      if (_selectedFilter == 'All') return true;
      if (_selectedFilter == 'Online') {
        return camera['status'] == 'Online';
      }
      if (_selectedFilter == 'Offline') {
        return camera['status'] == 'Offline';
      }

      return camera['location'] == _selectedFilter;
    }).toList();
  }

  int get _onlineCount =>
      _cameras.where((camera) => camera['status'] == 'Online').length;

  int get _offlineCount =>
      _cameras.where((camera) => camera['status'] == 'Offline').length;

  int get _recordingCount =>
      _cameras.where((camera) => camera['recording'] == true).length;

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: OwnerFeatureFrame(
          child: CustomScrollView(
            physics: const BouncingScrollPhysics(),
            slivers: [
              SliverToBoxAdapter(child: _buildHeader()),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildSummary()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 22, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildFilters()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(child: _buildSectionHeader()),
              ),
              _buildCameraGrid(),
              const SliverToBoxAdapter(child: SizedBox(height: 36)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF073F42), Color(0xFF0A696D), Color(0xFF0A797B)],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: _goBackToDashboard,
                  ),
                  const Spacer(),
                  _circleButton(
                    icon: Icons.more_horiz_rounded,
                    onTap: _showMoreMenu,
                  ),
                ],
              ),
              const SizedBox(height: 22),
              Text(
                'CCTV Monitoring',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 29,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 5),
              Text(
                widget.hostelName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 12.5,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.18)),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
      ),
    );
  }

  void _goBackToDashboard() {
    if (widget.onBackToDashboard != null) {
      widget.onBackToDashboard!();
      return;
    }
    Navigator.of(context).maybePop();
  }

  Widget _buildSummary() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.035),
            blurRadius: 16,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Camera Overview',
            style: GoogleFonts.dmSans(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _summaryItem(
                  value: '${_cameras.length}',
                  label: 'Total',
                  icon: Icons.videocam_rounded,
                  color: AppTheme.teal,
                ),
              ),
              _divider(),
              Expanded(
                child: _summaryItem(
                  value: '$_onlineCount',
                  label: 'Online',
                  icon: Icons.wifi_rounded,
                  color: const Color(0xFF3D7B53),
                ),
              ),
              _divider(),
              Expanded(
                child: _summaryItem(
                  value: '$_offlineCount',
                  label: 'Offline',
                  icon: Icons.wifi_off_rounded,
                  color: const Color(0xFFB14646),
                ),
              ),
              _divider(),
              Expanded(
                child: _summaryItem(
                  value: '$_recordingCount',
                  label: 'Recording',
                  icon: Icons.fiber_manual_record_rounded,
                  color: const Color(0xFFC43E3E),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _summaryItem({
    required String value,
    required String label,
    required IconData icon,
    required Color color,
  }) {
    return Column(
      children: [
        Icon(icon, size: 18, color: color),
        const SizedBox(height: 6),
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 17,
            fontWeight: FontWeight.w700,
            color: OwnerFeaturePalette.text,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 9.5,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _divider() {
    return Container(width: 1, height: 42, color: OwnerFeaturePalette.border);
  }

  Widget _buildFilters() {
    return SizedBox(
      height: 38,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _filters.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final filter = _filters[index];
          final selected = filter == _selectedFilter;

          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedFilter = filter;
              });
            },
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: selected ? AppTheme.teal : OwnerFeaturePalette.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: selected ? AppTheme.teal : OwnerFeaturePalette.border,
                ),
              ),
              child: Text(
                filter,
                style: GoogleFonts.dmSans(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: selected
                      ? Colors.white
                      : OwnerFeaturePalette.textMuted,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSectionHeader() {
    return Row(
      children: [
        Expanded(
          child: Text(
            'Live Cameras',
            style: GoogleFonts.playfairDisplay(
              fontSize: 19,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
        ),
        Text(
          '${_filteredCameras.length} cameras',
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _buildCameraGrid() {
    final cameras = _filteredCameras;

    if (cameras.isEmpty) {
      return SliverPadding(
        padding: const EdgeInsets.symmetric(horizontal: 18),
        sliver: SliverToBoxAdapter(child: _buildEmptyState()),
      );
    }

    return SliverPadding(
      padding: const EdgeInsets.symmetric(horizontal: 18),
      sliver: SliverGrid(
        delegate: SliverChildBuilderDelegate((context, index) {
          return _cameraCard(cameras[index]);
        }, childCount: cameras.length),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          mainAxisSpacing: 12,
          crossAxisSpacing: 12,
          childAspectRatio: 0.86,
        ),
      ),
    );
  }

  Widget _cameraCard(Map<String, dynamic> camera) {
    final online = camera['status'] == 'Online';
    final recording = camera['recording'] == true;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => CameraViewScreen(
                hostelName: widget.hostelName,
                camera: camera,
              ),
            ),
          );
        },
        child: Container(
          decoration: BoxDecoration(
            color: OwnerFeaturePalette.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    borderRadius: const BorderRadius.vertical(
                      top: Radius.circular(19),
                    ),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: online
                          ? [const Color(0xFF15383A), const Color(0xFF081F20)]
                          : [const Color(0xFF555A5B), const Color(0xFF2D3031)],
                    ),
                  ),
                  child: Stack(
                    children: [
                      Center(
                        child: Container(
                          width: 58,
                          height: 58,
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.07),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            camera['icon'] as IconData,
                            size: 28,
                            color: Colors.white.withOpacity(0.78),
                          ),
                        ),
                      ),

                      Positioned(
                        top: 10,
                        left: 10,
                        child: _cameraStatusBadge(
                          online ? 'LIVE' : 'OFFLINE',
                          online
                              ? const Color(0xFFD9534F)
                              : const Color(0xFF737373),
                        ),
                      ),

                      Positioned(
                        top: 10,
                        right: 10,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 7,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.28),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text(
                            camera['quality'].toString(),
                            style: GoogleFonts.dmSans(
                              fontSize: 8.5,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),

                      if (recording)
                        Positioned(
                          bottom: 10,
                          right: 10,
                          child: Row(
                            children: [
                              Container(
                                width: 7,
                                height: 7,
                                decoration: const BoxDecoration(
                                  color: Color(0xFFD9534F),
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 4),
                              Text(
                                'REC',
                                style: GoogleFonts.dmSans(
                                  fontSize: 8,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                ),
              ),

              Padding(
                padding: const EdgeInsets.fromLTRB(12, 11, 12, 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      camera['name'].toString(),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: OwnerFeaturePalette.text,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      camera['location'].toString(),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        fontSize: 9.5,
                        color: OwnerFeaturePalette.textMuted,
                      ),
                    ),
                    const SizedBox(height: 7),
                    Row(
                      children: [
                        Container(
                          width: 7,
                          height: 7,
                          decoration: BoxDecoration(
                            color: online
                                ? const Color(0xFF3D7B53)
                                : const Color(0xFFB14646),
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 5),
                        Expanded(
                          child: Text(
                            online
                                ? 'Streaming now'
                                : 'Last seen ${camera['lastUpdated']}',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(
                              fontSize: 8.5,
                              color: OwnerFeaturePalette.textMuted,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _cameraStatusBadge(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.90),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        label,
        style: GoogleFonts.dmSans(
          fontSize: 8,
          fontWeight: FontWeight.w800,
          color: Colors.white,
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 42),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Container(
            width: 70,
            height: 70,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.videocam_off_outlined,
              size: 31,
              color: AppTheme.teal,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'No cameras found',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Try changing the selected camera filter.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              fontSize: 11.5,
              height: 1.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  void _showMoreMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _menuTile(
                  icon: Icons.add_a_photo_outlined,
                  title: 'Add Camera',
                  onTap: () {
                    Navigator.pop(context);
                    _showTemporary(
                      'Camera configuration will be connected later.',
                    );
                  },
                ),
                _menuTile(
                  icon: Icons.cloud_done_outlined,
                  title: 'Recording Storage',
                  onTap: () {
                    Navigator.pop(context);
                    _showTemporary(
                      'Recording storage settings will be connected later.',
                    );
                  },
                ),
                _menuTile(
                  icon: Icons.security_outlined,
                  title: 'CCTV Permissions',
                  onTap: () {
                    Navigator.pop(context);
                    _showTemporary(
                      'CCTV access permissions will be connected later.',
                    );
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _menuTile({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      leading: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.07),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: AppTheme.teal, size: 19),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: OwnerFeaturePalette.text,
        ),
      ),
      trailing: const Icon(Icons.chevron_right_rounded),
    );
  }

  void _showTemporary(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(color: Colors.white),
          ),
        ),
      );
  }
}
